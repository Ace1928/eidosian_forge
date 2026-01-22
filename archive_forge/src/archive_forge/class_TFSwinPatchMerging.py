from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
class TFSwinPatchMerging(keras.layers.Layer):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`keras.layer.Layer`, *optional*, defaults to `keras.layers.LayerNormalization`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: Optional[Callable]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = keras.layers.Dense(2 * dim, use_bias=False, name='reduction')
        if norm_layer is None:
            self.norm = keras.layers.LayerNormalization(epsilon=1e-05, name='norm')
        else:
            self.norm = norm_layer(name='norm')

    def maybe_pad(self, input_feature: tf.Tensor, height: int, width: int) -> tf.Tensor:
        should_pad = height % 2 == 1 or width % 2 == 1
        if should_pad:
            pad_values = ((0, 0), (0, height % 2), (0, width % 2), (0, 0))
            input_feature = tf.pad(input_feature, pad_values)
        return input_feature

    def call(self, input_feature: tf.Tensor, input_dimensions: Tuple[int, int], training: bool=False) -> tf.Tensor:
        height, width = input_dimensions
        batch_size, _, num_channels = shape_list(input_feature)
        input_feature = tf.reshape(input_feature, (batch_size, height, width, num_channels))
        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = tf.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = tf.reshape(input_feature, (batch_size, -1, 4 * num_channels))
        input_feature = self.norm(input_feature, training=training)
        input_feature = self.reduction(input_feature, training=training)
        return input_feature

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'reduction', None) is not None:
            with tf.name_scope(self.reduction.name):
                self.reduction.build([None, None, 4 * self.dim])
        if getattr(self, 'norm', None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, 4 * self.dim])