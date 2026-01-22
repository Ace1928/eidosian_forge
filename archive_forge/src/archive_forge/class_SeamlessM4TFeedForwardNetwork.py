import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
class SeamlessM4TFeedForwardNetwork(nn.Module):

    def __init__(self, config: SeamlessM4TConfig, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.fc2.weight, torch.Tensor) and hidden_states.dtype != self.fc2.weight.dtype and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states