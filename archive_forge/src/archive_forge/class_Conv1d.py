import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.ao.nn.intrinsic import _FusedModule
from typing import Tuple, TypeVar, Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
class Conv1d(_ConvNd, nn.Conv1d):
    """
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as :class:`~torch.nn.Conv1d`

    Similar to :class:`~torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Conv1d
    _FLOAT_CONV_MODULE = nn.Conv1d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t=1, padding: Union[str, _size_1_t]=0, dilation: _size_1_t=1, groups: int=1, bias: bool=True, padding_mode: str='zeros', qconfig=None, device=None, dtype=None) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size_, stride=stride_, padding=padding_, dilation=dilation_, transposed=False, output_padding=_single(0), groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig, device=device, dtype=dtype)

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)