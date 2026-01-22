from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetShortCut(keras.layers.Layer):
    """
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=2, **kwargs):
        super().__init__(**kwargs)
        self.convolution = keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=stride, use_bias=False, name='convolution')
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name='normalization')
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, inputs: tf.Tensor, training: bool=False) -> tf.Tensor:
        return self.normalization(self.convolution(inputs), training=training)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution', None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, 'normalization', None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])