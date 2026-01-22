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
class SeamlessM4TVariancePredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=1)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        hidden_states = self.conv2(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(dim=2)