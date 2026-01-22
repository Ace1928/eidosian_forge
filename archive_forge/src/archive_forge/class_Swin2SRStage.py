import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
class Swin2SRStage(nn.Module):
    """
    This corresponds to the Residual Swin Transformer Block (RSTB) in the original implementation.
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, pretrained_window_size=0):
        super().__init__()
        self.config = config
        self.dim = dim
        self.layers = nn.ModuleList([Swin2SRLayer(config=config, dim=dim, input_resolution=input_resolution, num_heads=num_heads, shift_size=0 if i % 2 == 0 else config.window_size // 2, pretrained_window_size=pretrained_window_size) for i in range(depth)])
        if config.resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif config.resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.patch_embed = Swin2SRPatchEmbeddings(config, normalize_patches=False)
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        residual = hidden_states
        height, width = input_dimensions
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
        output_dimensions = (height, width, height, width)
        hidden_states = self.patch_unembed(hidden_states, input_dimensions)
        hidden_states = self.conv(hidden_states)
        hidden_states, _ = self.patch_embed(hidden_states)
        hidden_states = hidden_states + residual
        stage_outputs = (hidden_states, output_dimensions)
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs