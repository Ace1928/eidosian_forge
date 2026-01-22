import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
class QuantLinear(nn.Module):
    """
    Quantized version of `torch.nn.Linear`. Adds quantization-specific arguments on top of `torch.nn.Linear`.

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the quantized weight.
        bias_bit (`int`, *optional*, defaults to `32`):
            Bitwidth for the quantized bias.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether or not to use channel-wise quantization.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, in_features, out_features, bias=True, weight_bit=8, bias_bit=32, per_channel=False, quant_mode=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros([out_features, in_features]))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quant_mode = quant_mode
        self.percentile_mode = False
        self.weight_function = SymmetricQuantFunction.apply

    def __repr__(self):
        s = super().__repr__()
        s = f'({s} weight_bit={self.weight_bit}, quant_mode={self.quant_mode})'
        return s

    def forward(self, x, prev_act_scaling_factor=None):
        if not self.quant_mode:
            return (nn.functional.linear(x, weight=self.weight, bias=self.bias), None)
        assert prev_act_scaling_factor is not None and prev_act_scaling_factor.shape == (1,), 'Input activation to the QuantLinear layer should be globally (non-channel-wise) quantized. Please add a QuantAct layer with `per_channel = True` before this QuantAct layer'
        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)
        self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
        self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor)
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor
        if self.bias is not None:
            self.bias_integer = self.weight_function(self.bias, self.bias_bit, False, bias_scaling_factor)
        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor
        return (nn.functional.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) * bias_scaling_factor, bias_scaling_factor)