import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
class FunnelRelMultiheadAttention(nn.Module):

    def __init__(self, config: FunnelConfig, block_index: int) -> None:
        super().__init__()
        self.config = config
        self.block_index = block_index
        d_model, n_head, d_head = (config.d_model, config.n_head, config.d_head)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_head = nn.Linear(d_model, n_head * d_head)
        self.v_head = nn.Linear(d_model, n_head * d_head)
        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_r_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_kernel = nn.Parameter(torch.zeros([d_model, n_head, d_head]))
        self.r_s_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.seg_embed = nn.Parameter(torch.zeros([2, n_head, d_head]))
        self.post_proj = nn.Linear(n_head * d_head, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.scale = 1.0 / d_head ** 0.5

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        if self.config.attention_type == 'factorized':
            phi, pi, psi, omega = position_embeds
            u = self.r_r_bias * self.scale
            w_r = self.r_kernel
            q_r_attention = torch.einsum('binh,dnh->bind', q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]
            positional_attn = torch.einsum('bind,jd->bnij', q_r_attention_1, psi) + torch.einsum('bind,jd->bnij', q_r_attention_2, omega)
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            r = position_embeds[self.block_index][shift - 1]
            v = self.r_r_bias * self.scale
            w_r = self.r_kernel
            r_head = torch.einsum('td,dnh->tnh', r, w_r)
            positional_attn = torch.einsum('binh,tnh->bnit', q_head + v, r_head)
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)
        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        r_s_bias = self.r_s_bias * self.scale
        token_type_bias = torch.einsum('bind,snd->bnis', q_head + r_s_bias, self.seg_embed)
        token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.shape[2], seq_len, context_len])
        diff_token_type, same_token_type = torch.split(token_type_bias, 1, dim=-1)
        token_type_attn = torch.where(token_type_mat, same_token_type.expand(token_type_mat.shape), diff_token_type.expand(token_type_mat.shape))
        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_inputs: Tuple[torch.Tensor], output_attentions: bool=False) -> Tuple[torch.Tensor, ...]:
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = (self.config.n_head, self.config.d_head)
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, d_head)
        q_head = q_head * self.scale
        r_w_bias = self.r_w_bias * self.scale
        content_score = torch.einsum('bind,bjnd->bnij', q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)
        attn_score = content_score + positional_attn + token_type_attn
        dtype = attn_score.dtype
        attn_score = attn_score.float()
        if attention_mask is not None:
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
        attn_prob = self.attention_dropout(attn_prob)
        attn_vec = torch.einsum('bnij,bjnd->bind', attn_prob, v_head)
        attn_out = self.post_proj(attn_vec.reshape(batch_size, seq_len, n_head * d_head))
        attn_out = self.hidden_dropout(attn_out)
        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)