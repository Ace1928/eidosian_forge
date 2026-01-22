from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTVisionTransformer(keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = TFGroupViTVisionEmbeddings(config, name='embeddings')
        self.encoder = TFGroupViTVisionEncoder(config, name='encoder')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.embed_dim = config.hidden_size

    def call(self, pixel_values: TFModelInputType, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(hidden_states=embedding_output, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = tf.math.reduce_mean(last_hidden_state, axis=1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

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
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.embed_dim])