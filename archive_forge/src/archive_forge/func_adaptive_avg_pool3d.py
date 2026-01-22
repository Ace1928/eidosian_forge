from typing import List, Optional
import warnings
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2
from .modules.utils import _pair_from_first
def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    """
    Applies a 3D adaptive average pooling over a quantized input signal composed
    of several quantized input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
                     double-integer tuple)
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.functional.adaptive_avg_pool3d' must be quantized!")
    return torch.nn.functional.adaptive_avg_pool3d(input, output_size)