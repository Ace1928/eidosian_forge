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
class MaskedSoftmax(layers.Layer):
    """Performs a softmax with optional masking on a tensor.

    Args:
      mask_expansion_axes: Any axes that should be padded on the mask tensor.
      normalization_axes: On which axes the softmax should perform.
    """

    def __init__(self, mask_expansion_axes=None, normalization_axes=None, **kwargs):
        self._mask_expansion_axes = mask_expansion_axes
        if normalization_axes is None:
            self._normalization_axes = (-1,)
        else:
            self._normalization_axes = normalization_axes
        super(MaskedSoftmax, self).__init__(**kwargs)

    def call(self, scores, mask=None):
        if mask is not None:
            for _ in range(len(scores.shape) - len(mask.shape)):
                mask = tf.expand_dims(mask, axis=self._mask_expansion_axes)
            adder = (1.0 - tf.cast(mask, scores.dtype)) * -10000.0
            scores += adder
        if len(self._normalization_axes) == 1:
            return tf.nn.softmax(scores, axis=self._normalization_axes[0])
        else:
            return tf.math.exp(scores - tf.math.reduce_logsumexp(scores, axis=self._normalization_axes, keepdims=True))

    def get_config(self):
        config = {'mask_expansion_axes': self._mask_expansion_axes, 'normalization_axes': self._normalization_axes}
        base_config = super(MaskedSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))