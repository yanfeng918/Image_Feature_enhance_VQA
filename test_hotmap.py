import torch
import torchvision
import numpy as np
import cv2

# 加载预训练的 Faster R-CNN 模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 设置为评估模式
model.eval()

# 加载图片
image = cv2.imread('6.jpeg')

# 转换为 PyTorch Tensor
import torchvision.transforms.functional as F
image_tensor = F.to_tensor(image)

# 进行推理
output = model([image_tensor])

# 获取注意力热力图
attention_map = output[0]['scores'].cpu().detach().numpy()
attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map)) * 255
attention_map = attention_map.astype(np.uint8)
attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_COOL)

# 可视化注意力热力图
result = cv2.addWeighted(image, 0.7, attention_map, 0.3, 0)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

