from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_deit import DeiTConfig
class TFDeitPixelShuffle(keras.layers.Layer):
    """TF layer implementation of torch.nn.PixelShuffle"""

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f'upscale_factor must be an integer value >= 2 got {upscale_factor}')
        self.upscale_factor = upscale_factor

    def call(self, x: tf.Tensor) -> tf.Tensor:
        hidden_states = x
        batch_size, _, _, num_input_channels = shape_list(hidden_states)
        block_size_squared = self.upscale_factor ** 2
        output_depth = int(num_input_channels / block_size_squared)
        permutation = tf.constant([[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]])
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format='NHWC')
        return hidden_states