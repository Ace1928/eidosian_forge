from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import ops
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import FtrlParameters
from tensorflow.python.tpu.tpu_embedding import MomentumParameters
from tensorflow.python.tpu.tpu_embedding import ProximalAdagradParameters
from tensorflow.python.tpu.tpu_embedding import RMSPropParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _get_weights_from_features(weight_key_name, features):
    """Pop and return feature for weights, possibly None."""
    weights = None
    if weight_key_name is not None:
        if weight_key_name in features:
            weights = features.pop(weight_key_name)
        else:
            raise ValueError('Cannot find weights {} for weighted_categorical_column. Please check if the weights are present in feature dict. Also note weight-sharing among weighted_categorical_column is not supported on TPU.'.format(weight_key_name))
        if not isinstance(weights, tf.sparse.SparseTensor):
            raise ValueError('weighted_categorical_column with weight key name {} has dense weights. Dense weights are not supported on TPU. Please use sparse weights instead.'.format(weight_key_name))
        if weights.dtype is not tf.dtypes.float32:
            weights = tf.cast(weights, dtype=tf.dtypes.float32)
    return weights