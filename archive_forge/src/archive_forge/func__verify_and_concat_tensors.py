from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.saving import serialization_lib
def _verify_and_concat_tensors(self, output_tensors):
    """Verifies and concatenates the dense output of several columns."""
    _verify_static_batch_size_equality(output_tensors, self._feature_columns)
    return tf.concat(output_tensors, -1)