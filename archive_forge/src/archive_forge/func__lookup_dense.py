import collections
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
def _lookup_dense(self, inputs):
    """Lookup table values for a dense Tensor, handling masking and OOV."""
    if tf.executing_eagerly() and backend.is_keras_tensor(inputs):
        lookups = tf.zeros_like(inputs, dtype=self._value_dtype)
    else:
        lookups = self.lookup_table.lookup(inputs)
    if self.mask_token is not None:
        mask_locations = tf.equal(inputs, self._mask_key)
        lookups = tf.where(mask_locations, self._mask_value, lookups)
    if self.invert:
        return lookups
    lookup_checks = []
    if self.num_oov_indices == 0:
        oov_indices = tf.where(tf.equal(lookups, -1))
        oov_inputs = tf.gather_nd(inputs, oov_indices)
        msg = tf.strings.format('When `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.', (oov_inputs,))
        assertion = tf.Assert(tf.equal(tf.size(oov_indices), 0), [msg])
        lookup_checks.append(assertion)
    elif self.num_oov_indices > 1:
        if self._key_dtype.is_integer:
            oov_indices = tf.math.floormod(inputs, self.num_oov_indices)
        else:
            oov_indices = tf.strings.to_hash_bucket_fast(inputs, num_buckets=self.num_oov_indices)
        oov_indices = oov_indices + self._oov_start_index()
        oov_locations = tf.equal(lookups, self._default_value)
        lookups = tf.where(oov_locations, oov_indices, lookups)
    with tf.control_dependencies(lookup_checks):
        return tf.identity(lookups)