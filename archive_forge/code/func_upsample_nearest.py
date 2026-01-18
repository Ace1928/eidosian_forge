from typing import List, Optional
import warnings
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2
from .modules.utils import _pair_from_first
def upsample_nearest(input, size=None, scale_factor=None):
    """Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(..., mode='nearest')``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatial
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    warnings.warn('nn.quantized.functional.upsample_nearest is deprecated. Use nn.quantized.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode='nearest')