from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_convbert import ConvBertConfig
@keras_serializable
class TFConvBertMainLayer(keras.layers.Layer):
    config_class = ConvBertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = TFConvBertEmbeddings(config, name='embeddings')
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = keras.layers.Dense(config.hidden_size, name='embeddings_project')
        self.encoder = TFConvBertEncoder(config, name='encoder')
        self.config = config

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        hidden_states = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, hidden_states.dtype)
        head_mask = self.get_head_mask(head_mask)
        if hasattr(self, 'embeddings_project'):
            hidden_states = self.embeddings_project(hidden_states, training=training)
        hidden_states = self.encoder(hidden_states, extended_attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=training)
        return hidden_states

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
        if getattr(self, 'embeddings_project', None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                self.embeddings_project.build([None, None, self.config.embedding_size])