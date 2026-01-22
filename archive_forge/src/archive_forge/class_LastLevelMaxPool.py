from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation
from ..utils import _log_api_usage_once
class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d (not actual max_pool2d, we just subsample) on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append('pool')
        x.append(F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0))
        return (x, names)