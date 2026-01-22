from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
@keras_serializable
class TFLxmertMainLayer(keras.layers.Layer):
    config_class = LxmertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFLxmertEmbeddings(config, name='embeddings')
        self.encoder = TFLxmertEncoder(config, name='encoder')
        self.pooler = TFLxmertPooler(config, name='pooler')
        self.config = config

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, visual_feats=None, visual_pos=None, attention_mask=None, visual_attention_mask=None, token_type_ids=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if visual_pos is None or visual_feats is None:
            raise ValueError("visual_feats and visual_pos cannot be `None` in LXMERT's `call` method.")
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        embedding_output = self.embeddings(input_ids, token_type_ids, inputs_embeds, training)
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        if visual_attention_mask is not None:
            extended_visual_attention_mask = tf.reshape(visual_attention_mask, (input_shape[0], 1, 1, input_shape[1]))
            extended_visual_attention_mask = tf.expand_dims(tf.expand_dims(visual_attention_mask, axis=1), axis=1)
            extended_visual_attention_mask = tf.cast(extended_visual_attention_mask, dtype=embedding_output.dtype)
            extended_visual_attention_mask = tf.multiply(tf.subtract(one_cst, extended_visual_attention_mask), ten_thousand_cst)
        else:
            extended_visual_attention_mask = None
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, visual_feats, visual_pos, extended_visual_attention_mask, output_attentions, training)
        visual_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        vision_hidden_states = visual_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]
        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            vision_attentions = visual_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (language_attentions, vision_attentions, cross_encoder_attentions)
        hidden_states = (language_hidden_states, vision_hidden_states) if output_hidden_states else ()
        visual_output = vision_hidden_states[-1]
        lang_output = language_hidden_states[-1]
        pooled_output = self.pooler(lang_output)
        if not return_dict:
            return (lang_output, visual_output, pooled_output) + hidden_states + all_attentions
        return TFLxmertModelOutput(pooled_output=pooled_output, language_output=lang_output, vision_output=visual_output, language_hidden_states=language_hidden_states if output_hidden_states else None, vision_hidden_states=vision_hidden_states if output_hidden_states else None, language_attentions=language_attentions if output_attentions else None, vision_attentions=vision_attentions if output_attentions else None, cross_encoder_attentions=cross_encoder_attentions if output_attentions else None)

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