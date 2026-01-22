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
def _append_composite_tensor(target, to_append):
    """Helper function to append composite tensors to each other in the 0 axis.

  In order to support batching within a fit/evaluate/predict call, we need
  to be able to aggregate within a CompositeTensor. Unfortunately, the CT
  API currently does not make this easy - especially in V1 mode, where we're
  working with CompositeTensor Value objects that have no connection with the
  CompositeTensors that created them.

  Args:
    target: CompositeTensor or CompositeTensor value object that will be
      appended to.
    to_append: CompositeTensor or CompositeTensor value object to append to.
      'target'.

  Returns:
    A CompositeTensor or CompositeTensor value object.

  Raises:
    RuntimeError: if concatenation is not possible.
  """
    if type(target) is not type(to_append):
        raise RuntimeError('Unable to concatenate %s and %s' % (type(target), type(to_append)))
    if isinstance(target, sparse_tensor.SparseTensor):
        return sparse_ops.sparse_concat(sp_inputs=[target, to_append], axis=0)
    elif isinstance(target, ragged_tensor.RaggedTensor):
        return array_ops.concat([target, to_append], axis=0)
    elif isinstance(target, sparse_tensor.SparseTensorValue):
        return _append_sparse_tensor_value(target, to_append)
    elif isinstance(target, ragged_tensor_value.RaggedTensorValue):
        return _append_ragged_tensor_value(target, to_append)
    else:
        raise RuntimeError('Attempted to concatenate unsupported object %s.' % type(target))