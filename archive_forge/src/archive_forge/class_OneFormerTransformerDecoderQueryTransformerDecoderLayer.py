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
class OneFormerTransformerDecoderQueryTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        q = k = self.with_pos_embed(output, query_pos)
        output2 = self.self_attn(q, k, value=output, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout1(output2)
        output = self.norm1(output)
        output2 = self.multihead_attn(query=self.with_pos_embed(output, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(output2)
        output = self.norm3(output)
        return output

    def forward_pre(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        output2 = self.norm1(output)
        q = k = self.with_pos_embed(output2, query_pos)
        output2 = self.self_attn(q, k, value=output2, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout1(output2)
        output2 = self.norm2(output)
        output2 = self.multihead_attn(query=self.with_pos_embed(output2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout2(output2)
        output2 = self.norm3(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        output = output + self.dropout3(output2)
        return output

    def forward(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(output, memory, output_mask, memory_mask, output_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(output, memory, output_mask, memory_mask, output_key_padding_mask, memory_key_padding_mask, pos, query_pos)