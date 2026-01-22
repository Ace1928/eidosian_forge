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
@staticmethod
def _adjust_widths_groups_compatibilty(stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]) -> Tuple[List[int], List[int]]:
    """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
    widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
    group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]
    ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
    stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
    return (stage_widths, group_widths_min)