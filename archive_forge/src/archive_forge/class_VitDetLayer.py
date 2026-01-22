import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
class VitDetLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: VitDetConfig, drop_path_rate: float=0, window_size: int=0, use_residual_block: bool=False) -> None:
        super().__init__()
        dim = config.hidden_size
        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = VitDetAttention(config, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.drop_path = VitDetDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = VitDetMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))
        self.window_size = window_size
        self.use_residual_block = use_residual_block
        if self.use_residual_block:
            self.residual = VitDetResBottleneckBlock(config=config, in_channels=dim, out_channels=dim, bottleneck_channels=dim // 2)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        shortcut = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.window_size > 0:
            height, width = (hidden_states.shape[1], hidden_states.shape[2])
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)
        self_attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)
        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))
        hidden_states = shortcut + self.drop_path(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        if self.use_residual_block:
            hidden_states = self.residual(hidden_states)
        outputs = (hidden_states,) + outputs
        return outputs