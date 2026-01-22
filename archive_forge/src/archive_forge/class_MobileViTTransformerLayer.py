import math
from typing import Dict, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilevit import MobileViTConfig
class MobileViTTransformerLayer(nn.Module):

    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.attention = MobileViTAttention(config, hidden_size)
        self.intermediate = MobileViTIntermediate(config, hidden_size, intermediate_size)
        self.output = MobileViTOutput(config, hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output