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
class OneFormerTransformerDecoderCrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, output, memory, memory_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        output2, attention_weights = self.multihead_attn(query=self.with_pos_embed(output, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output = output + self.dropout(output2)
        output = self.norm(output)
        return (output, attention_weights)

    def forward_pre(self, output, memory, memory_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        output2 = self.norm(output)
        output2, attention_weights = self.multihead_attn(query=self.with_pos_embed(output2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output = output + self.dropout(output2)
        return (output, attention_weights)

    def forward(self, output, memory, memory_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)