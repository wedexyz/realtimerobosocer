'''import torch
# Model
import numpy
import cv2
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5\\runs\\train\\exp\\weights\\best.pt') # or yolov5m, yolov5l, yolov5x, custom
#olors = np.random.uniform(0, 255, size=(len(1), 3))


frame = cv2.VideoCapture(0)
detections = model(frame[..., ::-1])
results = detections.pandas().xyxy[0].to_dict(orient="records")
for result in results:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                # Do whatever you want
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)'''

import numpy as np
import cv2
from sys import modules
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5\\runs\\train\\exp\\weights\\best.pt') # or yolov5m, yolov5l, yolov5x, custom
#colors = np.random.uniform(0, 255, size=(len(1), 3))

cap = cv2.VideoCapture(0)





while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame,(640,640))
    #print(frame.shape)
    detections = model(frame[..., ::-1])
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    print(results)

    for result in results:
        
                con = result['confidence']
                cs = result['name']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)
                cv2.putText(frame, str(cs) + " " , (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 2)

           

       
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #results.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


