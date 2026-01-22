from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch.nn as nn
from torch import Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .._utils import _ModelURLs
class Conv3DSimple(nn.Conv3d):

    def __init__(self, in_planes: int, out_planes: int, midplanes: Optional[int]=None, stride: int=1, padding: int=1) -> None:
        super().__init__(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3, 3), stride=stride, padding=padding, bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return (stride, stride, stride)