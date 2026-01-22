from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertVisualFeatureEncoder(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.visn_fc = keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='visn_fc')
        self.visn_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='visn_layer_norm')
        self.box_fc = keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='box_fc')
        self.box_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='box_layer_norm')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.feat_dim = config.visual_feat_dim
        self.pos_dim = config.visual_pos_dim
        self.config = config

    def call(self, visn_input, training=False):
        feats, boxes = visn_input
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2
        output = self.dropout(output, training=training)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'visn_fc', None) is not None:
            with tf.name_scope(self.visn_fc.name):
                self.visn_fc.build([None, None, self.feat_dim])
        if getattr(self, 'visn_layer_norm', None) is not None:
            with tf.name_scope(self.visn_layer_norm.name):
                self.visn_layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, 'box_fc', None) is not None:
            with tf.name_scope(self.box_fc.name):
                self.box_fc.build([None, None, self.pos_dim])
        if getattr(self, 'box_layer_norm', None) is not None:
            with tf.name_scope(self.box_layer_norm.name):
                self.box_layer_norm.build([None, None, self.config.hidden_size])