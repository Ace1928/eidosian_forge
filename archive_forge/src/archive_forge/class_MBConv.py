import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
class MBConv(nn.Module):
    """MBConv: Mobile Inverted Residual Bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        p_stochastic_dropout (float): Probability of stochastic depth.
    """

    def __init__(self, in_channels: int, out_channels: int, expansion_ratio: float, squeeze_ratio: float, stride: int, activation_layer: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], p_stochastic_dropout: float=0.0) -> None:
        super().__init__()
        proj: Sequence[nn.Module]
        self.proj: nn.Module
        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()
        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)
        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode='row')
        else:
            self.stochastic_depth = nn.Identity()
        _layers = OrderedDict()
        _layers['pre_norm'] = norm_layer(in_channels)
        _layers['conv_a'] = Conv2dNormActivation(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, activation_layer=activation_layer, norm_layer=norm_layer, inplace=None)
        _layers['conv_b'] = Conv2dNormActivation(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, activation_layer=activation_layer, norm_layer=norm_layer, groups=mid_channels, inplace=None)
        _layers['squeeze_excitation'] = SqueezeExcitation(mid_channels, sqz_channels, activation=nn.SiLU)
        _layers['conv_c'] = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H / stride, W / stride].
        """
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x