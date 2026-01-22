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
class SeamlessM4Tv2TextToUnitDecoderLayer(nn.Module):

    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        self.dropout = config.dropout
        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4Tv2Attention(embed_dim=self.embed_dim, num_heads=decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding='same')
        self.activation_fn = ACT2FN[config.activation_function]
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding='same')
        self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
        self.conv_dropout = nn.Dropout(self.dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*
                or 0 for *masked*
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.transpose(1, 2)).transpose(1, 2)
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv2(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.conv_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.conv_layer_norm(hidden_states)
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += self_attn_weights
        return outputs