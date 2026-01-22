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
class SSDLiteFeatureExtractorMobileNet(nn.Module):

    def __init__(self, backbone: nn.Module, c4_pos: int, norm_layer: Callable[..., nn.Module], width_mult: float=1.0, min_depth: int=16):
        super().__init__()
        _log_api_usage_once(self)
        if backbone[c4_pos].use_res_connect:
            raise ValueError('backbone[c4_pos].use_res_connect should be False')
        self.features = nn.Sequential(nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]), nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1:]))
        get_depth = lambda d: max(min_depth, int(d * width_mult))
        extra = nn.ModuleList([_extra_block(backbone[-1].out_channels, get_depth(512), norm_layer), _extra_block(get_depth(512), get_depth(256), norm_layer), _extra_block(get_depth(256), get_depth(256), norm_layer), _extra_block(get_depth(256), get_depth(128), norm_layer)])
        _normal_init(extra)
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)
        for block in self.extra:
            x = block(x)
            output.append(x)
        return OrderedDict([(str(i), v) for i, v in enumerate(output)])