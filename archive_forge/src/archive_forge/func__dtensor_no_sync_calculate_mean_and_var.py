import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _dtensor_no_sync_calculate_mean_and_var(self, inputs, reduction_axes, keep_dims, mask=None):
    replica_tensor = _expand_tensor_with_local_replica_group(inputs)
    local_batch_size = tf.shape(replica_tensor)[1]
    updated_reduction_axes = [n + 1 for n in reduction_axes]
    if mask is None:
        mean, var = tf.nn.moments(replica_tensor, updated_reduction_axes, keepdims=keep_dims)
    else:
        mask_weights = tf.cast(mask, self.compute_dtype, name='mask_weights')
        mask_weights = tf.expand_dims(mask_weights, axis=-1, name='mask_weights_broadcasted')
        mask_weights = _expand_tensor_with_local_replica_group(mask_weights)
        mean, var = tf.nn.weighted_moments(replica_tensor, axes=updated_reduction_axes, frequency_weights=mask_weights, keepdims=keep_dims)
    mean = tf.repeat(mean, local_batch_size, axis=0)
    var = tf.repeat(var, local_batch_size, axis=0)
    if not keep_dims:
        for dim in reduction_axes[1:]:
            mean = tf.expand_dims(mean, axis=dim)
            var = tf.expand_dims(var, axis=dim)
    return (mean, var)