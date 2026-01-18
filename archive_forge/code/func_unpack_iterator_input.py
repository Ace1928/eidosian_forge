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
def unpack_iterator_input(iterator):
    """Convert a dataset iterator to a tuple of tensors `x, y, sample_weights`.

  Args:
    iterator: Instance of a dataset iterator.

  Returns:
    Tuple of tensors `x, y, weights`. `y` and `weights` entry may be None.
  """
    try:
        next_element = iterator.get_next()
    except errors.OutOfRangeError:
        raise RuntimeError('Your dataset iterator ran out of data; Make sure that your dataset can generate required number of samples.')
    if isinstance(next_element, (list, tuple)):
        if len(next_element) not in [2, 3]:
            raise ValueError('Please provide model inputs as a list or tuple of 2 or 3 elements: (input, target) or (input, target, sample_weights) Received %s' % next_element)
        if len(next_element) == 2:
            x, y = next_element
            weights = None
        else:
            x, y, weights = next_element
    else:
        x = next_element
        y = None
        weights = None
    return (x, y, weights)