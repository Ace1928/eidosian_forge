import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...ops import boxes as box_ops
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..vgg import VGG, vgg16, VGG16_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
    """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
    num_blocks = len(self.module_list)
    if idx < 0:
        idx += num_blocks
    out = x
    for i, module in enumerate(self.module_list):
        if i == idx:
            out = module(x)
    return out