#include <opencv2/opencv.hpp>
#include <iostream>
#include <dnn.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;


int inpWidth = 416;    
int inpHeight = 416;
float confThreshold = 0.5; 
float nmsThreshold = 0.4;

std::vector<std::string> classes;

int POSE_PAIRS[3][20][2] = {
	{   // COCO body
		{ 1,2 },{ 1,5 },{ 2,3 },
		{ 3,4 },{ 5,6 },{ 6,7 },
		{ 1,8 },{ 8,9 },{ 9,10 },
		{ 1,11 },{ 11,12 },{ 12,13 },
		{ 1,0 },{ 0,14 },
		{ 14,16 },{ 0,15 },{ 15,17 }
	},
	{   // MPI body
		{ 0,1 },{ 1,2 },{ 2,3 },
		{ 3,4 },{ 1,5 },{ 5,6 },
		{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
		{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
	},
	{   // hand
		{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
		{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
		{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
		{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
		{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
	} };

bool getOutputsNames(const Net& net, vector<String> &outstrings);
void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

int yolov3()
{
	VideoCapture y_vc("test.mp4");
	if (!y_vc.isOpened()) return -1;

	string classesFile = "coco.names";
	String yolov3_model = "yolov3.cfg";
	String weights = "yolov3.weights";
	ifstream classNamesFile(classesFile.c_str());
	if (classNamesFile.is_open())
	{
		string className = "";
		while (getline(classNamesFile, className)) {
			classes.push_back(className);
		}
	}
	else {
		std::cout << "can not open classNamesFile" << std::endl;
	}

	Net net = readNetFromDarknet(yolov3_model, weights);
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);
	cv::Mat frame;

	while (1)
	{
		y_vc >> frame;

		if (frame.empty()) {
			cout << "frame is empty!!!" << endl;
			return -1;
		}

		Mat blob;
		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		net.setInput(blob);

		vector<cv::Mat> outs;
		static vector<String> outstrings;
		getOutputsNames(net, outstrings);
		net.forward(outs, outstrings);

		postprocess(frame, outs);

		imshow("frame", frame);

		if (waitKey(10) == 27)
		{
			break;
		}
	}

	return 0;

}

bool getOutputsNames(const Net& net, vector<String> &outstrings)
{
	if (outstrings.empty())
	{
		vector<int> outLayers = net.getUnconnectedOutLayers();

		vector<cv::String> layersNames = net.getLayerNames();

		outstrings.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			outstrings[i] = layersNames[outLayers[i] - 1];
	}
	return true;
}

void postprocess(Mat& frame, vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	else
	{
		cout << "classes is empty..." << endl;
	}

	//绘制标签
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
	top = std::max(top, labelSize.height);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

int openpose()
{

	
	String modelTxt = "openpose_pose_coco.prototxt";
	String modelBin = "caffe_models\\pose\\coco\\pose_iter_440000.caffemodel";

	Net net = readNetFromCaffe(modelTxt, modelBin);

	int W_in = 368;
	int H_in = 368;
	float thresh = 0.1;

	VideoCapture cap;
	cap.open("test.mp4");

	if (!cap.isOpened())return -1;

	while (1) {

		Mat frame;

		cap >> frame;

		if (frame.empty()) {
			cout << "frame is empty!!!" << endl;
			return -1;
		}

		//创建输入
		Mat inputBlob = blobFromImage(frame, 1.0 / 255, Size(W_in, H_in), Scalar(0, 0, 0), false, false);

		//输入
		net.setInput(inputBlob);

		//得到网络输出结果，结果为热力图
		Mat result = net.forward();

		int midx, npairs;
		int H = result.size[2];
		int W = result.size[3];

		//得到检测结果的关键点点数
		int nparts = result.size[1];


		// find out, which model we have
		//判断输出的模型类别
		if (nparts == 19)
		{   // COCO body
			midx = 0;
			npairs = 17;
			nparts = 18; // skip background
		}
		else if (nparts == 16)
		{   // MPI body
			midx = 1;
			npairs = 14;
		}
		else if (nparts == 22)
		{   // hand
			midx = 2;
			npairs = 20;
		}
		else
		{
			cerr << "there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has " << nparts << " parts." << endl;
			return (0);
		}

		// 获得身体各部分坐标
		vector<Point> points(22);
		for (int n = 0; n < nparts; n++)
		{
			// Slice heatmap of corresponding body's part.
			Mat heatMap(H, W, CV_32F, result.ptr(0, n));
			// 找到最大值的点
			Point p(-1, -1), pm;
			double conf;
			minMaxLoc(heatMap, 0, &conf, 0, &pm);
			//判断置信度
			if (conf > thresh) {
				p = pm;
			}
			points[n] = p;
		}

		//连接身体各个部分，并且绘制
		float SX = float(frame.cols) / W;
		float SY = float(frame.rows) / H;
		for (int n = 0; n < npairs; n++)
		{
			Point2f a = points[POSE_PAIRS[midx][n][0]];
			Point2f b = points[POSE_PAIRS[midx][n][1]];

			//如果前一个步骤没有找到相应的点，则跳过
			if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
				continue;

			// 缩放至图像的尺寸
			a.x *= SX; a.y *= SY;
			b.x *= SX; b.y *= SY;

			//绘制
			line(frame, a, b, Scalar(0, 200, 0), 2);
			circle(frame, a, 3, Scalar(0, 0, 200), -1);
			circle(frame, b, 3, Scalar(0, 0, 200), -1);
		}

		imshow("frame", frame);
		waitKey(30);
	}
	return 0;
}

int main()
{
	//yolov3();
	openpose();
}