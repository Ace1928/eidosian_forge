import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_fsmt import FSMTConfig
class FSMTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: FSMTConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        embed_dim = embed_tokens.embedding_dim
        self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.embed_tokens.weight, modifier_rank=None):
                embed_tokens_weight_shape = self.embed_tokens.weight.shape
        else:
            embed_tokens_weight_shape = self.embed_tokens.weight.shape
        self.output_projection = nn.Linear(embed_tokens_weight_shape[1], embed_tokens_weight_shape[0], bias=False)
        self.output_projection.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_padding_mask: torch.Tensor, decoder_padding_mask: torch.Tensor, decoder_causal_mask: torch.Tensor, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        """
        Includes several features from "Jointly Learning to Align and Translate with Transformer Models" (Garg et al.,
        EMNLP 2019).

        Args:
            input_ids (`torch.LongTensor` of shape `(batch, tgt_len)`):
                previous decoder outputs for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            past_key_values (dict or None): dictionary used for storing state during generation
            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        Returns:
            BaseModelOutputWithPast or tuple:

                - the decoder's features of shape *(batch, tgt_len, embed_dim)*
                - the cache
                - hidden states
                - attentions
        """
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            positions = self.embed_positions(input_ids)
            if use_cache:
                input_ids = input_ids[:, -1:]
                positions = positions[:, -1:]
            x = self.embed_tokens(input_ids) * self.embed_scale
        elif inputs_embeds is not None:
            position_ids = inputs_embeds[:, :, 0].masked_fill(inputs_embeds[:, :, 0].eq(0), self.embed_positions.padding_idx)
            positions = self.embed_positions(position_ids)
            x = inputs_embeds * self.embed_scale
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        x += positions
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        next_decoder_cache = []
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                assert attn_mask.size()[0] == len(self.layers), f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                x = x.transpose(0, 1)
                all_hidden_states += (x,)
                x = x.transpose(0, 1)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            layer_state = past_key_values[idx] if past_key_values is not None else None
            x, layer_self_attn, layer_past, layer_cross_attn = decoder_layer(x, encoder_hidden_states, encoder_attn_mask=encoder_padding_mask, decoder_padding_mask=decoder_padding_mask, layer_state=layer_state, causal_mask=decoder_causal_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, output_attentions=output_attentions)
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if output_attentions:
                all_self_attns += (layer_self_attn,)
                all_cross_attns += (layer_cross_attn,)
        if output_hidden_states:
            x = x.transpose(0, 1)
            all_hidden_states += (x,)
            x = x.transpose(0, 1)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        x = self.output_projection(x)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [x, next_cache, all_hidden_states, all_self_attns, all_cross_attns] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attns)