import collections
import math
import os
import re
import unicodedata
from typing import List
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import constants
from autokeras.utils import data_utils
@keras.utils.register_keras_serializable()
class SelfAttentionMask(layers.Layer):
    """official.nlp.modeling.layers.SelfAttentionMask"""

    def call(self, inputs):
        from_tensor = inputs[0]
        to_mask = inputs[1]
        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_shape = get_shape_list(to_mask, expected_rank=2)
        to_seq_length = to_shape[1]
        to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=from_tensor.dtype)
        broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=from_tensor.dtype)
        mask = broadcast_ones * to_mask
        return mask

    def get_config(self):
        return super().get_config()