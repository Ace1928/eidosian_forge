import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
class ClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """

    def __init__(self, config: ClapAudioConfig):
        super().__init__()
        img_size = (config.spec_size, config.spec_size) if isinstance(config.spec_size, int) else config.spec_size
        patch_size = (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        patch_stride = (config.patch_stride, config.patch_stride) if isinstance(config.patch_stride, int) else config.patch_stride
        self.img_size = img_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = config.flatten_patch_embeds
        self.enable_fusion = config.enable_fusion
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)
        scale_factor = 4 if self.enable_fusion and config.fusion_type == 'channel_map' else 1
        self.proj = nn.Conv2d(config.patch_embed_input_channels * scale_factor, config.patch_embeds_hidden_size, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = nn.LayerNorm(config.patch_embeds_hidden_size) if config.enable_patch_layer_norm else nn.Identity()
        if self.enable_fusion:
            self.fusion_model = ClapAudioAFFBlock(config)
            self.mel_conv2d = nn.Conv2d(config.patch_embed_input_channels, config.patch_embeds_hidden_size, kernel_size=(patch_size[0], patch_size[1] * 3), stride=(patch_stride[0], patch_stride[1] * 3), padding=padding)

    def forward(self, hidden_states, is_longer_idx=None):
        if self.enable_fusion:
            global_hidden_states = hidden_states[:, 0:1, :, :]
            batch_size, num_channels, height, width = global_hidden_states.shape
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
            global_hidden_states = self.proj(global_hidden_states)
            output_width = global_hidden_states.size(-1)
            if len(is_longer_idx) > 0:
                local_hidden_states = hidden_states[is_longer_idx, 1:, :, :].contiguous()
                batch_size, num_channels, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size * num_channels, 1, height, width)
                local_hidden_states = self.mel_conv2d(local_hidden_states)
                _, features, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size, num_channels, features, height, width)
                local_hidden_states = local_hidden_states.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)
                local_width = local_hidden_states.size(-1)
                local_hidden_states = torch.nn.functional.pad(local_hidden_states, (0, output_width - local_width), 'constant', 0)
                global_hidden_states[is_longer_idx] = self.fusion_model(global_hidden_states[is_longer_idx], local_hidden_states)
            hidden_states = global_hidden_states
        else:
            _, _, height, width = hidden_states.shape
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
            hidden_states = self.proj(hidden_states)
        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        return hidden_states