import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from torch.nn.common_types import _size_1_t
from .utils import ReferenceQuantizedModule
class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    """ A reference version of nn.quantized.ConvTranspose2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        qref_conv = cls(float_conv.in_channels, float_conv.out_channels, float_conv.kernel_size, float_conv.stride, float_conv.padding, float_conv.output_padding, float_conv.groups, float_conv.bias is not None, float_conv.dilation, float_conv.padding_mode, device=float_conv.weight.device, dtype=float_conv.weight.dtype, weight_qparams=weight_qparams)
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv