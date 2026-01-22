from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
@keras_serializable
class TFLongformerMainLayer(keras.layers.Layer):
    config_class = LongformerConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, '`config.attention_window` has to be an even value'
            assert config.attention_window > 0, '`config.attention_window` has to be positive'
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}'
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.pad_token_id = config.pad_token_id
        self.attention_window = config.attention_window
        self.embeddings = TFLongformerEmbeddings(config, name='embeddings')
        self.encoder = TFLongformerEncoder(config, name='encoder')
        self.pooler = TFLongformerPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, head_mask=None, global_attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if input_ids is not None and (not isinstance(input_ids, tf.Tensor)):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
        elif input_ids is not None:
            input_ids = tf.cast(input_ids, tf.int64)
        if attention_mask is not None and (not isinstance(attention_mask, tf.Tensor)):
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
        elif attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.int64)
        if global_attention_mask is not None and (not isinstance(global_attention_mask, tf.Tensor)):
            global_attention_mask = tf.convert_to_tensor(global_attention_mask, dtype=tf.int64)
        elif global_attention_mask is not None:
            global_attention_mask = tf.cast(global_attention_mask, tf.int64)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.cast(tf.fill(input_shape, 1), tf.int64)
        if token_type_ids is None:
            token_type_ids = tf.cast(tf.fill(input_shape, 0), tf.int64)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, pad_token_id=self.pad_token_id)
        is_index_masked = tf.math.less(attention_mask, 1)
        is_index_global_attn = tf.math.greater(attention_mask, 1)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)
        attention_mask_shape = shape_list(attention_mask)
        extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], attention_mask_shape[1], 1, 1))
        extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0
        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, padding_len=padding_len, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFLongformerBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, global_attentions=encoder_outputs.global_attentions)

    def _pad_to_window_size(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, pad_token_id):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        attention_window = self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])
        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)
        if position_ids is not None:
            position_ids = tf.pad(position_ids, paddings, constant_values=pad_token_id)
        if inputs_embeds is not None:
            if padding_len > 0:
                input_ids_padding = tf.cast(tf.fill((batch_size, padding_len), self.pad_token_id), tf.int64)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)
        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)
        token_type_ids = tf.pad(token_type_ids, paddings, constant_values=0)
        return (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)

    @staticmethod
    def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)