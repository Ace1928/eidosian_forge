import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTextTransformerLayer(nn.Module):

    def __init__(self, width: int, heads: int, attn_mask: torch.Tensor, layer_norm_eps=1e-05):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads)
        self.layer_norm1 = nn.LayerNorm(width, eps=layer_norm_eps)
        self.mlp = OneFormerTextMLP(width, width * 4, width)
        self.layer_norm2 = nn.LayerNorm(width, eps=layer_norm_eps)
        self.attn_mask = attn_mask

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states, need_weights=False, key_padding_mask=key_padding_mask)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states