import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
class Wav2Vec2BertAdapterLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.output_hidden_size
        dropout = config.conformer_conv_dropout
        self.kernel_size = config.adapter_kernel_size
        self.stride = config.adapter_stride
        self.residual_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.residual_conv = nn.Conv1d(embed_dim, 2 * embed_dim, self.kernel_size, stride=self.stride, padding=self.stride // 2)
        self.activation = nn.GLU(dim=1)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn_conv = nn.Conv1d(embed_dim, 2 * embed_dim, self.kernel_size, stride=self.stride, padding=self.stride // 2)
        self.self_attn = Wav2Vec2BertSelfAttention(config, is_adapter_attention=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn = Wav2Vec2BertFeedForward(config, act_fn=config.adapter_act, hidden_size=embed_dim)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, sub_sampled_lengths: Optional[torch.Tensor]=None):
        residual = self.residual_layer_norm(hidden_states)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        residual = residual.transpose(1, 2)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        hidden_states, attn_weigths = self.self_attn(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual
        return hidden_states