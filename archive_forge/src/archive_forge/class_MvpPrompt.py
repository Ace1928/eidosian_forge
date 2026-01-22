import copy
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mvp import MvpConfig
class MvpPrompt(nn.Module):
    """Layer-wise prompt for encoder or decoder."""

    def __init__(self, config, num_layers, num_heads):
        super().__init__()
        self.prompt_length = config.prompt_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = config.d_model // num_heads
        self.dropout = nn.Dropout(p=config.dropout)
        self.prompt_embedding = nn.Embedding(config.prompt_length, config.d_model)
        self.prompt_trans = nn.Sequential(nn.Linear(config.d_model, config.prompt_mid_dim), nn.GELU(), nn.Linear(config.prompt_mid_dim, num_layers * 2 * config.d_model))

    def forward(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        prompt = self.prompt_trans(self.prompt_embedding(prompt_ids))
        prompt = prompt.view(self.prompt_length, self.num_layers * 2, self.num_heads, self.head_dim)
        prompt = self.dropout(prompt)
        prompt = prompt.permute([1, 2, 0, 3]).split(2)
        return prompt