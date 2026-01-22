from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
class TFSegformerDWConv(keras.layers.Layer):

    def __init__(self, dim: int=768, **kwargs):
        super().__init__(**kwargs)
        self.depthwise_convolution = keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding='same', groups=dim, name='dwconv')
        self.dim = dim

    def call(self, hidden_states: tf.Tensor, height: int, width: int) -> tf.Tensor:
        batch_size = shape_list(hidden_states)[0]
        num_channels = shape_list(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
        hidden_states = self.depthwise_convolution(hidden_states)
        new_height = shape_list(hidden_states)[1]
        new_width = shape_list(hidden_states)[2]
        num_channels = shape_list(hidden_states)[3]
        hidden_states = tf.reshape(hidden_states, (batch_size, new_height * new_width, num_channels))
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'depthwise_convolution', None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                self.depthwise_convolution.build([None, None, None, self.dim])