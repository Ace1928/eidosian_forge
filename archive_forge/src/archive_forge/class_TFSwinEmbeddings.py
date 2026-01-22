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
class TFSwinEmbeddings(keras.layers.Layer):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: SwinConfig, use_mask_token: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.patch_embeddings = TFSwinPatchEmbeddings(config, name='patch_embeddings')
        self.num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.embed_dim = config.embed_dim
        self.use_mask_token = use_mask_token
        self.use_absolute_embeddings = config.use_absolute_embeddings
        self.norm = keras.layers.LayerNormalization(name='norm', epsilon=1e-05)
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')
        self.config = config

    def build(self, input_shape: tf.TensorShape) -> None:
        if self.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.embed_dim), initializer='zeros', name='mask_token')
        else:
            self.mask_token = None
        if self.use_absolute_embeddings:
            self.position_embeddings = self.add_weight((1, self.num_patches + 1, self.embed_dim), initializer='zeros', name='positional_embeddings')
        else:
            self.position_embeddings = None
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        if getattr(self, 'norm', None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.embed_dim])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: bool=None, training: bool=False) -> Tuple[tf.Tensor, Tuple[int, int]]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values, training=training)
        embeddings = self.norm(embeddings, training=training)
        batch_size, seq_len, _ = shape_list(embeddings)
        if bool_masked_pos is not None:
            mask_tokens = tf.repeat(self.mask_token, batch_size, 0)
            mask_tokens = tf.repeat(mask_tokens, seq_len, 1)
            mask = tf.expand_dims(bool_masked_pos, -1)
            mask = tf.cast(mask, mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return (embeddings, output_dimensions)