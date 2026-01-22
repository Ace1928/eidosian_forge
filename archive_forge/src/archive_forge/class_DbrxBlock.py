import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
class DbrxBlock(nn.Module):

    def __init__(self, config: DbrxConfig, block_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        self.norm_attn_norm = DbrxNormAttentionNorm(config=config, block_idx=block_idx)
        self.ffn = DbrxFFN(config=config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: torch.LongTensor=None, past_key_value: Optional[Cache]=None, output_attentions: Optional[bool]=False, output_router_logits: Optional[bool]=False, use_cache: Optional[bool]=False, cache_position: Optional[torch.LongTensor]=None, **kwargs: Any) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[Cache]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[Cache], Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]]]:
        """Forward function for DbrxBlock.

        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            position_ids (`torch.LongTensor`): position ids of shape `(batch, seq_len)`
            attention_mask (`torch.Tensor`, optional): attention mask of size (batch_size, sequence_length)
                if flash attention is used or (batch_size, 1, query_sequence_length, key_sequence_length)
                if default attention is used.
            past_key_value (`Tuple(torch.Tensor)`, optional): cached past key and value projection states
            output_attentions (`bool`, optional): Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under returned tensors for more detail.
            output_router_logits (`bool`, optional): Whether or not to return the router logits.
            use_cache (`bool`, optional): If set to `True`, `past_key_values` key value states are
                returned and can be used to speed up decoding (see `past_key_values`).
            cache_position (`torch.LongTensor`, optional): position ids of the cache
        """
        resid_states, hidden_states, self_attn_weights, present_key_value = self.norm_attn_norm(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, **kwargs)
        hidden_states, router_logits = self.ffn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = resid_states + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs