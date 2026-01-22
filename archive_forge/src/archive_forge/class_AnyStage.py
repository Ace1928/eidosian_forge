import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, width_in: int, width_out: int, stride: int, depth: int, block_constructor: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]=None, stage_index: int=0) -> None:
        super().__init__()
        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, stride if i == 0 else 1, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio)
            self.add_module(f'block{stage_index}-{i}', block)