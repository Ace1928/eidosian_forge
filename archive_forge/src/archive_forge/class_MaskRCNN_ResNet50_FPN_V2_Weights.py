from collections import OrderedDict
from typing import Any, Callable, Optional
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead
class MaskRCNN_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth', transforms=ObjectDetection, meta={**_COMMON_META, 'num_params': 46359409, 'recipe': 'https://github.com/pytorch/vision/pull/5773', '_metrics': {'COCO-val2017': {'box_map': 47.4, 'mask_map': 41.8}}, '_ops': 333.577, '_file_size': 177.219, '_docs': 'These weights were produced using an enhanced training recipe to boost the model accuracy.'})
    DEFAULT = COCO_V1