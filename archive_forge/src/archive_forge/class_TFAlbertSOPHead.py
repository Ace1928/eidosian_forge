from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_albert import AlbertConfig
class TFAlbertSOPHead(keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)
        self.dropout = keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.config = config

    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor:
        dropout_pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=dropout_pooled_output)
        return logits

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])