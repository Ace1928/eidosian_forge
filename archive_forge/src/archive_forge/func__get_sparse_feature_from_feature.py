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
def _get_sparse_feature_from_feature(feature_key, features):
    """Pop and return sparse feature."""
    sparse_feature = features.pop(feature_key)
    if not sparse_feature.dtype.is_integer:
        raise ValueError('SparseTensor with string as values are not supported. If you are using categorical_column_with_vocabulary_file or categorical_column_with_vocabulary_list, please call your_column.categorical_column._transform_feature({{your_column.key: features[your_column.key]}}) in your input_fn() to convert string to int. feature_key = {}.'.format(feature_key))
    return sparse_feature