import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientnet import EfficientNetConfig
class EfficientNetDepthwiseLayer(nn.Module):
    """
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(self, config: EfficientNetConfig, in_dim: int, stride: int, kernel_size: int, adjust_padding: bool):
        super().__init__()
        self.stride = stride
        conv_pad = 'valid' if self.stride == 2 else 'same'
        padding = correct_pad(kernel_size, adjust=adjust_padding)
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        self.depthwise_conv = EfficientNetDepthwiseConv2d(in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False)
        self.depthwise_norm = nn.BatchNorm2d(num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)
        return hidden_states