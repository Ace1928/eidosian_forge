import collections
import math
import string
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.layers import activation
from keras.src.layers import core
from keras.src.layers import regularization
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _build_attention(self, rank):
    """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
            rank: the rank of query, key, value tensors.
        """
    if self._attention_axes is None:
        self._attention_axes = tuple(range(1, rank - 2))
    else:
        self._attention_axes = tuple(self._attention_axes)
    self._dot_product_equation, self._combine_equation, attn_scores_rank = _build_attention_equation(rank, attn_axes=self._attention_axes)
    norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = activation.Softmax(axis=norm_axes, dtype=self._dtype_policy)
    self._dropout_layer = regularization.Dropout(rate=self._dropout, dtype=self._dtype_policy)