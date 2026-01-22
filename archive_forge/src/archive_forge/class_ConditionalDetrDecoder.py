import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_conditional_detr import ConditionalDetrConfig
class ConditionalDetrDecoder(ConditionalDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`ConditionalDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for Conditional DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: ConditionalDetrConfig
    """

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.layers = nn.ModuleList([ConditionalDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)
        d_model = config.d_model
        self.gradient_checkpointing = False
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        for layer_id in range(config.decoder_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        self.post_init()

    def forward(self, inputs_embeds=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, object_queries=None, query_position_embeddings=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        position_embeddings = kwargs.pop('position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        intermediate = () if self.config.auxiliary_loss else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        reference_points_before_sigmoid = self.ref_point_head(query_position_embeddings)
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        obj_center = reference_points[..., :2].transpose(0, 1)
        query_sine_embed_before_transformation = gen_sine_position_embeddings(obj_center, self.config.d_model)
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            if idx == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(hidden_states)
            query_sine_embed = query_sine_embed_before_transformation * pos_transformation
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, None, object_queries, query_position_embeddings, query_sine_embed, encoder_hidden_states, encoder_attention_mask, None, None)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=None, object_queries=object_queries, query_position_embeddings=query_position_embeddings, query_sine_embed=query_sine_embed, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, is_first=idx == 0)
            hidden_states = layer_outputs[0]
            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate, reference_points] if v is not None))
        return ConditionalDetrDecoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions, intermediate_hidden_states=intermediate, reference_points=reference_points)