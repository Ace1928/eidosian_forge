import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
class SwiftFormerConvEncoder(nn.Module):
    """
    `SwiftFormerConvEncoder` with 3*3 and 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * dim)
        self.depth_wise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.point_wise_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.point_wise_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.depth_wise_conv(x)
        x = self.norm(x)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        x = input + self.drop_path(self.layer_scale * x)
        return x