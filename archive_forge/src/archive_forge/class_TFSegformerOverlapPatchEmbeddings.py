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
class TFSegformerOverlapPatchEmbeddings(keras.layers.Layer):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.proj = keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=stride, padding='VALID', name='proj')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')
        self.num_channels = num_channels
        self.hidden_size = hidden_size

    def call(self, pixel_values: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        embeddings = self.proj(self.padding(pixel_values))
        height = shape_list(embeddings)[1]
        width = shape_list(embeddings)[2]
        hidden_dim = shape_list(embeddings)[3]
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        embeddings = self.layer_norm(embeddings)
        return (embeddings, height, width)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'proj', None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, None, self.num_channels])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])