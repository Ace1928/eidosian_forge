import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig
class FocalNetLayer(nn.Module):
    """Focal Modulation Network layer (block).

    Args:
        config (`FocalNetConfig`):
            Model config.
        index (`int`):
            Layer index.
        dim (`int`):
            Number of input channels.
        input_resolution (`Tuple[int]`):
            Input resulotion.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """

    def __init__(self, config, index, dim, input_resolution, drop_path=0.0):
        super().__init__()
        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.drop = config.hidden_dropout_prob
        self.use_post_layernorm = config.use_post_layernorm
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.modulation = FocalNetModulation(config=config, index=index, dim=dim, projection_dropout=self.drop)
        self.drop_path = FocalNetDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(dim * config.mlp_ratio)
        self.mlp = FocalNetMlp(config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=self.drop)
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if config.use_layerscale:
            self.gamma_1 = nn.Parameter(config.layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(config.layerscale_value * torch.ones(dim), requires_grad=True)

    def forward(self, hidden_state, input_dimensions):
        height, width = input_dimensions
        batch_size, _, num_channels = hidden_state.shape
        shortcut = hidden_state
        hidden_state = hidden_state if self.use_post_layernorm else self.norm1(hidden_state)
        hidden_state = hidden_state.view(batch_size, height, width, num_channels)
        hidden_state = self.modulation(hidden_state).view(batch_size, height * width, num_channels)
        hidden_state = hidden_state if not self.use_post_layernorm else self.norm1(hidden_state)
        hidden_state = shortcut + self.drop_path(self.gamma_1 * hidden_state)
        hidden_state = hidden_state + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(hidden_state)) if self.use_post_layernorm else self.mlp(self.norm2(hidden_state))))
        return hidden_state