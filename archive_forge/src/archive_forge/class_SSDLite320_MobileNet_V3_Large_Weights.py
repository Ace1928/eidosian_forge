import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .. import mobilenet
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .ssd import SSD, SSDScoringHead
class SSDLite320_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth', transforms=ObjectDetection, meta={'num_params': 3440060, 'categories': _COCO_CATEGORIES, 'min_size': (1, 1), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#ssdlite320-mobilenetv3-large', '_metrics': {'COCO-val2017': {'box_map': 21.3}}, '_ops': 0.583, '_file_size': 13.418, '_docs': 'These weights were produced by following a similar training recipe as on the paper.'})
    DEFAULT = COCO_V1