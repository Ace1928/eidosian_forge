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
class SwiftFormerEncoderBlock(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2)
    SwiftFormerEfficientAdditiveAttention, and (3) MLP block.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels,height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int, drop_path: float=0.0) -> None:
        super().__init__()
        layer_scale_init_value = config.layer_scale_init_value
        use_layer_scale = config.use_layer_scale
        self.local_representation = SwiftFormerLocalRepresentation(config, dim=dim)
        self.attn = SwiftFormerEfficientAdditiveAttention(config, dim=dim)
        self.linear = SwiftFormerMlp(config, in_features=dim)
        self.drop_path = SwiftFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        batch_size, channels, height, width = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))
        return x