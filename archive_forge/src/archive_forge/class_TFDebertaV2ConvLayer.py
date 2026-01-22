from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
class TFDebertaV2ConvLayer(keras.layers.Layer):

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = getattr(config, 'conv_kernel_size', 3)
        self.conv_act = get_tf_activation(getattr(config, 'conv_act', 'tanh'))
        self.padding = (self.kernel_size - 1) // 2
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name='dropout')
        self.config = config

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope('conv'):
            self.conv_kernel = self.add_weight(name='kernel', shape=[self.kernel_size, self.config.hidden_size, self.config.hidden_size], initializer=get_initializer(self.config.initializer_range))
            self.conv_bias = self.add_weight(name='bias', shape=[self.config.hidden_size], initializer=tf.zeros_initializer())
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    def call(self, hidden_states: tf.Tensor, residual_states: tf.Tensor, input_mask: tf.Tensor, training: bool=False) -> tf.Tensor:
        out = tf.nn.conv2d(tf.expand_dims(hidden_states, 1), tf.expand_dims(self.conv_kernel, 0), strides=1, padding=[[0, 0], [0, 0], [self.padding, self.padding], [0, 0]])
        out = tf.squeeze(tf.nn.bias_add(out, self.conv_bias), 1)
        rmask = tf.cast(1 - input_mask, tf.bool)
        out = tf.where(tf.broadcast_to(tf.expand_dims(rmask, -1), shape_list(out)), 0.0, out)
        out = self.dropout(out, training=training)
        out = self.conv_act(out)
        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)
        if input_mask is None:
            output_states = output
        else:
            if len(shape_list(input_mask)) != len(shape_list(layer_norm_input)):
                if len(shape_list(input_mask)) == 4:
                    input_mask = tf.squeeze(tf.squeeze(input_mask, axis=1), axis=1)
                input_mask = tf.cast(tf.expand_dims(input_mask, axis=2), tf.float32)
            output_states = output * input_mask
        return output_states