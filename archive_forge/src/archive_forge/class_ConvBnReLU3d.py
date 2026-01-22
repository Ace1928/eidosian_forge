import math
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from typing import TypeVar
class ConvBnReLU3d(ConvBn3d):
    """
    A ConvBnReLU3d module is a module fused from Conv3d, BatchNorm3d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBnReLU3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = nn.ReLU
    _FUSED_FLOAT_MODULE = nni.ConvReLU3d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, eps, momentum, freeze_bn, qconfig)

    def forward(self, input):
        return F.relu(ConvBn3d._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super().from_float(mod)