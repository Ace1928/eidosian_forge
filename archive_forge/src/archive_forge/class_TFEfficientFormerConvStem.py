import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerConvStem(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=1)
        self.convolution1 = keras.layers.Conv2D(filters=out_channels // 2, kernel_size=3, strides=2, padding='valid', name='convolution1')
        self.batchnorm_before = keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name='batchnorm_before')
        self.convolution2 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=2, padding='valid', name='convolution2')
        self.batchnorm_after = keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name='batchnorm_after')
        self.activation = keras.layers.Activation(activation=keras.activations.relu, name='activation')
        self.out_channels = out_channels
        self.config = config

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> tf.Tensor:
        features = self.batchnorm_before(self.convolution1(self.padding(pixel_values)), training=training)
        features = self.activation(features)
        features = self.batchnorm_after(self.convolution2(self.padding(features)), training=training)
        features = self.activation(features)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution1', None) is not None:
            with tf.name_scope(self.convolution1.name):
                self.convolution1.build([None, None, None, self.config.num_channels])
        if getattr(self, 'batchnorm_before', None) is not None:
            with tf.name_scope(self.batchnorm_before.name):
                self.batchnorm_before.build([None, None, None, self.out_channels // 2])
        if getattr(self, 'convolution2', None) is not None:
            with tf.name_scope(self.convolution2.name):
                self.convolution2.build([None, None, None, self.out_channels // 2])
        if getattr(self, 'batchnorm_after', None) is not None:
            with tf.name_scope(self.batchnorm_after.name):
                self.batchnorm_after.build([None, None, None, self.out_channels])
        if getattr(self, 'activation', None) is not None:
            with tf.name_scope(self.activation.name):
                self.activation.build(None)