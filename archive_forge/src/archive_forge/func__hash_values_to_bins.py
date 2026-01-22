import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def _hash_values_to_bins(self, values):
    """Converts a non-sparse tensor of values to bin indices."""
    hash_bins = self.num_bins
    mask = None
    if self.mask_value is not None and hash_bins > 1:
        hash_bins -= 1
        mask = tf.equal(values, self.mask_value)
    if values.dtype.is_integer:
        values = tf.as_string(values)
    if self.strong_hash:
        values = tf.strings.to_hash_bucket_strong(values, hash_bins, name='hash', key=self.salt)
    else:
        values = tf.strings.to_hash_bucket_fast(values, hash_bins, name='hash')
    if mask is not None:
        values = tf.add(values, tf.ones_like(values))
        values = tf.where(mask, tf.zeros_like(values), values)
    return values