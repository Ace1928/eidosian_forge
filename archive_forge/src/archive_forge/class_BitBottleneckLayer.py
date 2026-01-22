import collections
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig
class BitBottleneckLayer(nn.Module):
    """Non Pre-activation bottleneck block, equivalent to V1.5/V1b bottleneck. Used for ViT Hybrid."""

    def __init__(self, config, in_channels, out_channels=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1, drop_path_rate=0.0, is_first_layer=False):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_channels = out_channels or in_channels
        mid_chs = make_div(out_channels * bottle_ratio)
        if is_first_layer:
            self.downsample = BitDownsampleConv(config, in_channels, out_channels, stride=stride, preact=False)
        else:
            self.downsample = None
        self.conv1 = WeightStandardizedConv2d(in_channels, mid_chs, 1, eps=1e-08, padding=config.global_padding)
        self.norm1 = BitGroupNormActivation(config, num_channels=mid_chs)
        self.conv2 = WeightStandardizedConv2d(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups, eps=1e-08, padding=config.global_padding)
        self.norm2 = BitGroupNormActivation(config, num_channels=mid_chs)
        self.conv3 = WeightStandardizedConv2d(mid_chs, out_channels, 1, eps=1e-08, padding=config.global_padding)
        self.norm3 = BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)
        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        shortcut = hidden_states
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.activation(hidden_states + shortcut)
        return hidden_states