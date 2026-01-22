import collections
import math
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pvt import PvtConfig
class PvtLayer(nn.Module):

    def __init__(self, config: PvtConfig, hidden_size: int, num_attention_heads: int, drop_path: float, sequences_reduction_ratio: float, mlp_ratio: float):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = PvtAttention(config=config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequences_reduction_ratio=sequences_reduction_ratio)
        self.drop_path = PvtDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = PvtFFN(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool=False):
        self_attention_outputs = self.attention(hidden_states=self.layer_norm_1(hidden_states), height=height, width=width, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states))
        mlp_output = self.drop_path(mlp_output)
        layer_output = hidden_states + mlp_output
        outputs = (layer_output,) + outputs
        return outputs