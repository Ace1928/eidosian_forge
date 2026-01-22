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
class TFMobileViTDeepLabV3(keras.layers.Layer):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.aspp = TFMobileViTASPP(config, name='aspp')
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = TFMobileViTConvLayer(config, in_channels=config.aspp_out_channels, out_channels=config.num_labels, kernel_size=1, use_normalization=False, use_activation=False, bias=True, name='classifier')

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        features = self.aspp(hidden_states[-1], training=training)
        features = self.dropout(features, training=training)
        features = self.classifier(features, training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'aspp', None) is not None:
            with tf.name_scope(self.aspp.name):
                self.aspp.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)