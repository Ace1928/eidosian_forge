import copy
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_umt5 import UMT5Config
class UMT5LayerSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.SelfAttention = UMT5Attention(config, has_relative_attention_bias=True)
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, past_key_value=None):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, past_key_value=past_key_value)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs