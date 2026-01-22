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
def _split_tensor(tensor, num_splits):
    """Splits tensor into num_splits pieces, returns a list of pieces."""
    if tensor is None:
        return [None] * num_splits
    elif num_splits <= 0:
        return ValueError('Tensors cannot be split into {} pieces.'.format(num_splits))
    elif num_splits == 1:
        return [tensor]
    elif isinstance(tensor, tf.sparse.SparseTensor):
        return tf.compat.v2.sparse.split(tensor, num_splits, axis=0)
    else:
        return tf.split(tensor, num_splits)