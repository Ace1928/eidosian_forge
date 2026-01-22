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
class FocalNetModulation(nn.Module):

    def __init__(self, config, index, dim, focal_factor=2, bias=True, projection_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.focal_window = config.focal_windows[index]
        self.focal_level = config.focal_levels[index]
        self.focal_factor = focal_factor
        self.use_post_layernorm_in_modulation = config.use_post_layernorm_in_modulation
        self.normalize_modulator = config.normalize_modulator
        self.projection_in = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.projection_context = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.activation = nn.GELU()
        self.projection_out = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False), nn.GELU()))
            self.kernel_sizes.append(kernel_size)
        if self.use_post_layernorm_in_modulation:
            self.layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state:
                Input features with shape of (batch_size, height, width, num_channels)
        """
        num_channels = hidden_state.shape[-1]
        x = self.projection_in(hidden_state).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (num_channels, num_channels, self.focal_level + 1), 1)
        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, level:level + 1]
        ctx_global = self.activation(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        self.modulator = self.projection_context(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_post_layernorm_in_modulation:
            x_out = self.layernorm(x_out)
        x_out = self.projection_out(x_out)
        x_out = self.projection_dropout(x_out)
        return x_out