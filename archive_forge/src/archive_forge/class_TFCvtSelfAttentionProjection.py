from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
class TFCvtSelfAttentionProjection(keras.layers.Layer):
    """Convolutional Projection for Attention."""

    def __init__(self, config: CvtConfig, embed_dim: int, kernel_size: int, stride: int, padding: int, projection_method: str='dw_bn', **kwargs):
        super().__init__(**kwargs)
        if projection_method == 'dw_bn':
            self.convolution_projection = TFCvtSelfAttentionConvProjection(config, embed_dim, kernel_size, stride, padding, name='convolution_projection')
        self.linear_projection = TFCvtSelfAttentionLinearProjection()

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.convolution_projection(hidden_state, training=training)
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution_projection', None) is not None:
            with tf.name_scope(self.convolution_projection.name):
                self.convolution_projection.build(None)