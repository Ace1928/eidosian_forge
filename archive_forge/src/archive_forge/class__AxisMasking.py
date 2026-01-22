import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class _AxisMasking(torch.nn.Module):
    """Apply masking to a spectrogram.

    Args:
        mask_param (int): Maximum possible length of the mask.
        axis (int): What dimension the mask is applied on (assuming the tensor is 3D).
            For frequency masking, axis = 1.
            For time masking, axis = 2.
        iid_masks (bool): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the dimension of the input tensor is >= 3.
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)
    """
    __constants__ = ['mask_param', 'axis', 'iid_masks', 'p']

    def __init__(self, mask_param: int, axis: int, iid_masks: bool, p: float=1.0) -> None:
        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks
        self.p = p

    def forward(self, specgram: Tensor, mask_value: float=0.0) -> Tensor:
        """
        Args:
            specgram (Tensor): Tensor of dimension `(..., freq, time)`.
            mask_value (float): Value to assign to the masked columns.

        Returns:
            Tensor: Masked spectrogram of dimensions `(..., freq, time)`.
        """
        if self.iid_masks:
            return F.mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + specgram.dim() - 3, p=self.p)
        else:
            return F.mask_along_axis(specgram, self.mask_param, mask_value, self.axis + specgram.dim() - 3, p=self.p)