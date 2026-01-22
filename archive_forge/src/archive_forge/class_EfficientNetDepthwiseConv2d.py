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
class EfficientNetDepthwiseConv2d(nn.Conv2d):

    def __init__(self, in_channels, depth_multiplier=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        out_channels = in_channels * depth_multiplier
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)