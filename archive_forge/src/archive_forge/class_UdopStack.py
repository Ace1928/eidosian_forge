import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class UdopStack(UdopPreTrainedModel):
    """
    This class is based on `T5Stack`, but modified to take into account the image modality as well as 2D position
    embeddings.
    """

    def __init__(self, config, embed_tokens=None, embed_patches=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.embed_patches = embed_patches
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length
        self.num_layers = config.num_layers
        self.block = nn.ModuleList([UdopBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)])
        self.final_layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if not self.is_decoder:
            self.cell_2d_embedding = UdopCellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)
        self.relative_bias = self._get_relative_bias(config)
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias)

    @staticmethod
    def _get_relative_bias(config: UdopConfig) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, bbox=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, pixel_values=None, visual_bbox=None, image_embeddings=None, position_bias=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(f'You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time')
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and (torch.numel(input_ids) == 0):
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            bbox = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(self.get_extended_attention_mask(attention_mask, input_shape))
            logger.warning('Empty batch')
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(f'You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds')
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError('You have to intialize the model with valid token embeddings')
            inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None:
            image_embeddings = self.embed_patches(pixel_values)
        if image_embeddings is not None:
            num_patches = self.config.image_size // self.config.patch_size
            inputs_embeds, bbox, attention_mask = combine_image_text_embeddings(image_embeddings, inputs_embeds, bbox, visual_bbox, attention_mask, num_patches, 0, self.config.image_size, self.config.patch_size)
            input_shape = inputs_embeds.size()[:-1]
        if not self.is_decoder and bbox is not None:
            inputs_embeds += self.cell_2d_embedding(bbox)
        batch_size, seq_length = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            assert self.is_decoder, '`use_cache` can only be set to `True` if {} is used as a decoder'.format(self)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and (encoder_hidden_states is not None):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.is_decoder else None
        if self.is_decoder:
            position_bias = None
        else:
            position_bias = self.relative_bias(attention_mask=attention_mask, bbox=bbox)
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None
        hidden_states = inputs_embeds
        hidden_states = self.dropout(hidden_states)
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, layer_head_mask=head_mask[i], past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, attention_mask, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithAttentionMask(last_hidden_state=hidden_states, attention_mask=attention_mask, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)