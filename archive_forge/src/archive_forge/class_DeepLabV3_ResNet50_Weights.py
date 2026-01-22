from functools import partial
from typing import Any, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead
class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(url='https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth', transforms=partial(SemanticSegmentation, resize_size=520), meta={**_COMMON_META, 'num_params': 42004074, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50', '_metrics': {'COCO-val2017-VOC-labels': {'miou': 66.4, 'pixel_acc': 92.4}}, '_ops': 178.722, '_file_size': 160.515})
    DEFAULT = COCO_WITH_VOC_LABELS_V1