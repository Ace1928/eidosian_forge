from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamPositionalEmbedding(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.scale = config.hidden_size // 2
        self.config = config

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(name='positional_embedding', shape=(2, self.config.num_pos_feats), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=self.scale), trainable=False)
        super().build(input_shape)

    def call(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = tf.identity(input_coords)
        if input_shape is not None:
            coordinates = tf.stack([tf.cast(coordinates[:, :, :, 0], tf.float32) / input_shape[1], tf.cast(coordinates[:, :, :, 1], tf.float32) / input_shape[0]], axis=-1)
        coordinates = 2 * coordinates - 1
        coordinates = tf.cast(coordinates, self.positional_embedding.dtype)
        coordinates = tf.matmul(coordinates, self.positional_embedding)
        coordinates = 2 * np.pi * coordinates
        return tf.concat([tf.sin(coordinates), tf.cos(coordinates)], axis=-1)