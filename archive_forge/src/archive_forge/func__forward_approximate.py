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
def _forward_approximate(self, input):
    """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
    assert self.bn.running_var is not None
    running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
    scale_factor = self.bn.weight / running_std
    weight_shape = [1] * len(self.weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(self.weight.shape)
    bias_shape[1] = -1
    scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
    if self.bias is not None:
        zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
    else:
        zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
    conv = self._conv_forward(input, scaled_weight, zero_bias)
    conv_orig = conv / scale_factor.reshape(bias_shape)
    if self.bias is not None:
        conv_orig = conv_orig + self.bias.reshape(bias_shape)
    conv = self.bn(conv_orig)
    return conv