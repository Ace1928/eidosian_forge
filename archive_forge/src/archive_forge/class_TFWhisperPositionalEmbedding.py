from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
class TFWhisperPositionalEmbedding(keras.layers.Layer):

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int]=None, embedding_initializer=None, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_initializer = keras.initializers.get(embedding_initializer)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', shape=[self.num_positions, self.embedding_dim], initializer=self.embedding_initializer, trainable=True)
        super().build(input_shape)

    def call(self, input_ids, past_key_values_length=0):
        past_key_values_length = tf.cast(past_key_values_length, tf.int32)
        gather_indices = tf.range(tf.shape(input_ids)[1], delta=1) + past_key_values_length
        return tf.gather(self.weight, gather_indices)