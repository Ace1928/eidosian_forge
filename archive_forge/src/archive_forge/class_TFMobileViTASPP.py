from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTASPP(keras.layers.Layer):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels
        if len(config.atrous_rates) != 3:
            raise ValueError('Expected 3 values for atrous_rates')
        self.convs = []
        in_projection = TFMobileViTConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=1, use_activation='relu', name='convs.0')
        self.convs.append(in_projection)
        self.convs.extend([TFMobileViTConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=rate, use_activation='relu', name=f'convs.{i + 1}') for i, rate in enumerate(config.atrous_rates)])
        pool_layer = TFMobileViTASPPPooling(config, in_channels, out_channels, name=f'convs.{len(config.atrous_rates) + 1}')
        self.convs.append(pool_layer)
        self.project = TFMobileViTConvLayer(config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation='relu', name='project')
        self.dropout = keras.layers.Dropout(config.aspp_dropout_prob)

    def call(self, features: tf.Tensor, training: bool=False) -> tf.Tensor:
        features = tf.transpose(features, perm=[0, 2, 3, 1])
        pyramid = []
        for conv in self.convs:
            pyramid.append(conv(features, training=training))
        pyramid = tf.concat(pyramid, axis=-1)
        pooled_features = self.project(pyramid, training=training)
        pooled_features = self.dropout(pooled_features, training=training)
        return pooled_features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'project', None) is not None:
            with tf.name_scope(self.project.name):
                self.project.build(None)
        if getattr(self, 'convs', None) is not None:
            for conv in self.convs:
                with tf.name_scope(conv.name):
                    conv.build(None)