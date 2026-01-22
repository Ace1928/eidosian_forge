from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
class TFResNetConvLayer(keras.layers.Layer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, activation: str='relu', **kwargs) -> None:
        super().__init__(**kwargs)
        self.pad_value = kernel_size // 2
        self.conv = keras.layers.Conv2D(out_channels, kernel_size=kernel_size, strides=stride, padding='valid', use_bias=False, name='convolution')
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name='normalization')
        self.activation = ACT2FN[activation] if activation is not None else keras.layers.Activation('linear')
        self.in_channels = in_channels
        self.out_channels = out_channels

    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        height_pad = width_pad = (self.pad_value, self.pad_value)
        hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
        hidden_state = self.conv(hidden_state)
        return hidden_state

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv', None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, None, self.in_channels])
        if getattr(self, 'normalization', None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])