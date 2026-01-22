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
class FocalNetStage(nn.Module):

    def __init__(self, config, index, input_resolution):
        super().__init__()
        self.config = config
        self.num_stages = len(config.depths)
        embed_dim = [config.embed_dim * 2 ** i for i in range(self.num_stages)]
        dim = embed_dim[index]
        out_dim = embed_dim[index + 1] if index < self.num_stages - 1 else None
        downsample = FocalNetPatchEmbeddings if index < self.num_stages - 1 else None
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        drop_path = dpr[sum(config.depths[:index]):sum(config.depths[:index + 1])]
        self.layers = nn.ModuleList([FocalNetLayer(config=config, index=index, dim=dim, input_resolution=input_resolution, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(config.depths[index])])
        if downsample is not None:
            self.downsample = downsample(config=config, image_size=input_resolution, patch_size=2, num_channels=dim, embed_dim=out_dim, add_norm=True, use_conv_embed=config.use_conv_embed, is_stem=False)
        else:
            self.downsample = None
        self.pointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, input_dimensions)
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height, width = input_dimensions
            hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states_before_downsampling.shape[0], -1, height, width)
            hidden_states, output_dimensions = self.downsample(hidden_states)
        else:
            output_dimensions = (height, width, height, width)
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
        return stage_outputs