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
class AdaptiveAveragePooling1D(keras.layers.Layer):
    """
    Args:
    Average 1D Pooling with adaptive kernel size.
      output_size: An integer or tuple/list of a single integer, specifying pooled_features.
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, steps, channels)` while `channels_first` corresponds
        to inputs with shape `(batch, channels, steps)`.
    Input shape:
      - If `data_format='channels_last'`: 3D tensor with shape `(batch, steps, channels)`.
      - If `data_format='channels_first'`: 3D tensor with shape `(batch, channels, steps)`.
    Output shape:
      - If `data_format='channels_last'`: 3D tensor with shape `(batch_size, pooled_steps, channels)`.
      - If `data_format='channels_first'`: 3D tensor with shape `(batch_size, channels, pooled_steps)`.

    Adapted from [tensorflow-addon's adaptive pooling.py](
        https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/layers/adaptive_pooling.py#L90-L120
    )
    """

    def __init__(self, output_size: Union[int, Iterable[int]], reduce_function: Callable=tf.reduce_mean, data_format: Optional[str]=None, **kwargs) -> None:
        self.data_format = normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = (output_size,) if isinstance(output_size, int) else tuple(output_size)
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, *args) -> None:
        bins = self.output_size[0]
        if self.data_format == 'channels_last':
            splits = tf.split(inputs, bins, axis=1)
            splits = tf.stack(splits, axis=1)
            out_vect = self.reduce_function(splits, axis=2)
        else:
            splits = tf.split(inputs, bins, axis=2)
            splits = tf.stack(splits, axis=2)
            out_vect = self.reduce_function(splits, axis=3)
        return out_vect

    def compute_output_shape(self, input_shape: Iterable[int]) -> tf.TensorShape:
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            shape = tf.TensorShape([input_shape[0], self.output_size[0], input_shape[2]])
        else:
            shape = tf.TensorShape([input_shape[0], input_shape[1], self.output_size[0]])
        return shape

    def get_config(self) -> Dict[str, Any]:
        config = {'output_size': self.output_size, 'data_format': self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}