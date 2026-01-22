from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig
class RegNetXLayer(nn.Module):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        self.layer = nn.Sequential(RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act), RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act), RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None))
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state