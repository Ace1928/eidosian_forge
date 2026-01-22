import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
class QuantAct(nn.Module):
    """
    Quantizes the given activation.

    Args:
        activation_bit (`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.
        channel_len (`int`, *optional*):
            Specify the channel length when set the *per_channel* True.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit, act_range_momentum=0.95, per_channel=False, channel_len=None, quant_mode=False):
        super().__init__()
        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.percentile = False
        self.act_function = SymmetricQuantFunction.apply
        if not self.per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
            self.x_min -= 1e-05
            self.x_max += 1e-05
        else:
            raise NotImplementedError('per-channel mode is not currently supported for activation.')

    def __repr__(self):
        return f'{self.__class__.__name__}(activation_bit={self.activation_bit}, quant_mode: {self.quant_mode}, Act_min: {self.x_min.item():.2f}, Act_max: {self.x_max.item():.2f})'

    def forward(self, x, pre_act_scaling_factor=None, identity=None, identity_scaling_factor=None, specified_min=None, specified_max=None):
        x_act = x if identity is None else identity + x
        if self.training:
            assert not self.percentile, 'percentile mode is not currently supported for activation.'
            assert not self.per_channel, 'per-channel mode is not currently supported for activation.'
            x_min = x_act.data.min()
            x_max = x_act.data.max()
            assert x_max.isnan().sum() == 0 and x_min.isnan().sum() == 0, 'NaN detected when computing min/max of the activation'
            if self.x_min.min() > -1.1e-05 and self.x_max.max() < 1.1e-05:
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)
        if not self.quant_mode:
            return (x_act, None)
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max
        self.act_scaling_factor = symmetric_linear_quantization_params(self.activation_bit, x_min, x_max, per_channel=self.per_channel)
        if pre_act_scaling_factor is None:
            quant_act_int = self.act_function(x, self.activation_bit, self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = FixedPointMul.apply(x, pre_act_scaling_factor, self.activation_bit, self.act_scaling_factor, identity, identity_scaling_factor)
        correct_output_scale = self.act_scaling_factor.view(-1)
        return (quant_act_int * correct_output_scale, self.act_scaling_factor)