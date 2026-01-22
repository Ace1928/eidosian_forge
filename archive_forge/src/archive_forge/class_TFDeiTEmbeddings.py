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
class TFDeiTEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: DeiTConfig, use_mask_token: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.use_mask_token = use_mask_token
        self.patch_embeddings = TFDeiTPatchEmbeddings(config=config, name='patch_embeddings')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=keras.initializers.zeros(), trainable=True, name='cls_token')
        self.distillation_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=keras.initializers.zeros(), trainable=True, name='distillation_token')
        self.mask_token = None
        if self.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=keras.initializers.zeros(), trainable=True, name='mask_token')
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.add_weight(shape=(1, num_patches + 2, self.config.hidden_size), initializer=keras.initializers.zeros(), trainable=True, name='position_embeddings')
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None=None, training: bool=False) -> tf.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_length, _ = shape_list(embeddings)
        if bool_masked_pos is not None:
            mask_tokens = tf.tile(self.mask_token, [batch_size, seq_length, 1])
            mask = tf.expand_dims(bool_masked_pos, axis=-1)
            mask = tf.cast(mask, dtype=mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        distillation_tokens = tf.repeat(self.distillation_token, repeats=batch_size, axis=0)
        embeddings = tf.concat((cls_tokens, distillation_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return embeddings