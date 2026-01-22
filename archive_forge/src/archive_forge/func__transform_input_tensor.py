import abc
import collections
import math
import re
import numpy as np
import six
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _transform_input_tensor(self, input_tensor):
    """Returns a SparseTensor with identity values."""
    if not input_tensor.dtype.is_integer:
        raise ValueError('Invalid input, not integer. key: {} dtype: {}'.format(self.key, input_tensor.dtype))
    values = input_tensor.values
    if input_tensor.values.dtype != dtypes.int64:
        values = math_ops.cast(values, dtypes.int64, name='values')
    if self.default_value is not None:
        values = math_ops.cast(input_tensor.values, dtypes.int64, name='values')
        num_buckets = math_ops.cast(self.num_buckets, dtypes.int64, name='num_buckets')
        zero = math_ops.cast(0, dtypes.int64, name='zero')
        values = array_ops.where_v2(math_ops.logical_or(values < zero, values >= num_buckets, name='out_of_range'), array_ops.fill(dims=array_ops.shape(values), value=math_ops.cast(self.default_value, dtypes.int64), name='default_values'), values)
    return sparse_tensor_lib.SparseTensor(indices=input_tensor.indices, values=values, dense_shape=input_tensor.dense_shape)