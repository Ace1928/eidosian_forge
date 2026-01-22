import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _append_sparse_tensor_value(target, to_append):
    """Append sparse tensor value objects."""
    if len(target.dense_shape) != len(to_append.dense_shape):
        raise RuntimeError('Unable to concatenate %s and %s. The inner dense shapes do not have the same number of dimensions (%s vs %s)' % (target, to_append, target.dense_shape, to_append.dense_shape))
    if target.dense_shape[1:] != to_append.dense_shape[1:]:
        raise RuntimeError('Unable to concatenate %s and %s. The inner dense shapes do not match inner dimensions (%s vs %s)' % (target, to_append, target.dense_shape[1:], to_append.dense_shape[1:]))
    base_dim0_value = target.dense_shape[0]
    max_dim0_value = target.dense_shape[0]
    new_indices = target.indices
    for index in to_append.indices:
        index[0] += base_dim0_value
        max_dim0_value = max(max_dim0_value, index[0])
        new_indices = np.append(new_indices, [index], axis=0)
    new_values = np.concatenate((target.values, to_append.values), axis=0)
    new_dense_shape = list(target.dense_shape)
    new_dense_shape[0] = max_dim0_value + 1
    new_dense_shape = tuple(new_dense_shape)
    return sparse_tensor.SparseTensorValue(indices=new_indices, values=new_values, dense_shape=new_dense_shape)