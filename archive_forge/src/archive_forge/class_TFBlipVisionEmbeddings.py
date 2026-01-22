from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
class TFBlipVisionEmbeddings(keras.layers.Layer):

    def __init__(self, config: BlipVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = keras.layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, kernel_initializer=get_initializer(self.config.initializer_range), data_format='channels_last', name='patch_embedding')
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

    def build(self, input_shape=None):
        self.class_embedding = self.add_weight(shape=(1, 1, self.embed_dim), initializer=get_initializer(self.config.initializer_range), trainable=True, name='class_embedding')
        self.position_embedding = self.add_weight(shape=(1, self.num_positions, self.embed_dim), initializer=get_initializer(self.config.initializer_range), trainable=True, name='position_embedding')
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embedding', None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, 3])

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(pixel_values)[0]
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = tf.reshape(patch_embeds, (batch_size, self.num_patches, -1))
        class_embeds = tf.broadcast_to(self.class_embedding, (batch_size, 1, self.embed_dim))
        embeddings = tf.concat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding[:, :tf.shape(embeddings)[1], :]
        return embeddings