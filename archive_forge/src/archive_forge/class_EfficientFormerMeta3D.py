import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerMeta3D(nn.Module):

    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float=0.0):
        super().__init__()
        self.token_mixer = EfficientFormerSelfAttention(dim=config.dim, key_dim=config.key_dim, num_heads=config.num_attention_heads, attention_ratio=config.attention_ratio, resolution=config.resolution)
        self.layernorm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)
        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.token_mixer(self.layernorm1(hidden_states), output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * attention_output)
            layer_output = layer_output + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.layernorm2(layer_output)))
        else:
            layer_output = hidden_states + self.drop_path(attention_output)
            layer_output = layer_output + self.drop_path(self.mlp(self.layernorm2(layer_output)))
        outputs = (layer_output,) + outputs
        return outputs