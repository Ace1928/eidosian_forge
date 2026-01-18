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
def unpack_validation_data(validation_data, raise_if_ambiguous=True):
    """Unpack validation data based input type.

  The validation data is not touched if its dataset or dataset iterator.
  For other type of input (Numpy or tensor), it will be unpacked into tuple of
  3 which is x, y and sample weights.

  Args:
    validation_data: dataset, dataset iterator, or numpy, tensor tuple.
    raise_if_ambiguous: boolean on whether to fail if validation_data cannot be
      parsed. Otherwise simply return validation_data, None, None and defer the
      decision to the caller.

  Returns:
    tuple of 3, (x, y, sample_weights) for numpy and tensor input.
  """
    if isinstance(validation_data, (iterator_ops.Iterator, iterator_ops.IteratorBase, data_types.DatasetV2, data_utils.Sequence)) or not hasattr(validation_data, '__len__'):
        val_x = validation_data
        val_y = None
        val_sample_weight = None
    elif len(validation_data) == 2:
        try:
            val_x, val_y = validation_data
            val_sample_weight = None
        except ValueError:
            val_x, val_y, val_sample_weight = (validation_data, None, None)
    elif len(validation_data) == 3:
        try:
            val_x, val_y, val_sample_weight = validation_data
        except ValueError:
            val_x, val_y, val_sample_weight = (validation_data, None, None)
    else:
        if raise_if_ambiguous:
            raise ValueError('When passing a `validation_data` argument, it must contain either 2 items (x_val, y_val), or 3 items (x_val, y_val, val_sample_weights), or alternatively it could be a dataset or a dataset or a dataset iterator. However we received `validation_data=%s`' % validation_data)
        val_x, val_y, val_sample_weight = (validation_data, None, None)
    return (val_x, val_y, val_sample_weight)