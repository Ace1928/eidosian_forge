import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
class PegasusXEncoderLayer(nn.Module):

    def __init__(self, stagger_blocks_this_layer: bool, config: PegasusXConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = PegasusXGlobalLocalAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, block_size=config.block_size, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.global_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.stagger_blocks_this_layer = stagger_blocks_this_layer
        self.block_size = config.block_size

    def forward(self, hidden_states: torch.Tensor, global_hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool=False) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            global_hidden_states (`torch.FloatTensor`): global token hidden states
                *(seq_len, num_global_tokens, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        global_residual = global_hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        global_hidden_states = self.global_self_attn_layer_norm(global_hidden_states)
        if self.stagger_blocks_this_layer:
            hidden_states, attention_mask = self.pad_local_tokens(hidden_states=hidden_states, attention_mask=attention_mask, block_size=self.block_size)
        hidden_states, global_hidden_states, attn_weights = self.self_attn(token_hidden_states=hidden_states, global_hidden_states=global_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        if self.stagger_blocks_this_layer:
            hidden_states = self.unpad_local_tokens(padded_hidden_states=hidden_states, block_size=self.block_size)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.dropout, training=self.training)
        global_hidden_states = global_residual + global_hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        global_residual = global_hidden_states
        global_hidden_states = self.final_layer_norm(global_hidden_states)
        global_hidden_states = self.activation_fn(self.fc1(global_hidden_states))
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.activation_dropout, training=self.training)
        global_hidden_states = self.fc2(global_hidden_states)
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.dropout, training=self.training)
        global_hidden_states = global_residual + global_hidden_states
        outputs = (hidden_states, global_hidden_states)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    @classmethod
    def pad_local_tokens(cls, hidden_states, attention_mask, block_size):
        pad_size = block_size // 2
        mask_min_value = torch.finfo(hidden_states.dtype).min
        padded_hidden_states = torch.nn.functional.pad(hidden_states, pad=(0, 0, pad_size, pad_size))
        padded_mask = torch.nn.functional.pad(attention_mask, pad=(pad_size, pad_size), value=mask_min_value)
        return (padded_hidden_states, padded_mask)

    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size):
        pad_size = block_size // 2
        return padded_hidden_states[:, pad_size:-pad_size, :]