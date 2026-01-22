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
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
class SeamlessM4Tv2DecoderLayer(nn.Module):

    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4Tv2Attention(embed_dim=self.embed_dim, num_heads=decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = SeamlessM4Tv2Attention(self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True)
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`):
                encoder attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by
                very large negative values.
            past_key_value (`Tuple(torch.FloatTensor)`):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attention(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, past_key_value=cross_attn_past_key_value, attention_mask=encoder_attention_mask, output_attentions=output_attentions)
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states
            present_key_value += cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs