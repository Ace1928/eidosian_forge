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
def validate_dataset_input(x, y, sample_weight, validation_split=None):
    """Validates user input arguments when a dataset iterator is passed.

  Args:
    x: Input data. A `tf.data` dataset or iterator.
    y: Target data. It could be either Numpy array(s) or TensorFlow tensor(s).
      Expected to be `None` when `x` is a dataset iterator.
    sample_weight: An optional sample-weight array passed by the user to weight
      the importance of each sample in `x`. Expected to be `None` when `x` is a
      dataset iterator
    validation_split: Float between 0 and 1. Fraction of the training data to be
      used as validation data. Expected to be `None` when `x` is a dataset
      iterator.

  Raises:
    ValueError: if argument `y` or `sample_weight` or `validation_split` are
        provided by user.
  """
    if y is not None:
        raise ValueError('You passed a dataset or dataset iterator (%s) as input `x` to your model. In that case, you should not specify a target (`y`) argument, since the dataset or dataset iterator generates both input data and target data. Received: %s' % (x, y))
    if sample_weight is not None:
        raise ValueError('`sample_weight` argument is not supported when input `x` is a dataset or a dataset iterator. Instead, youcan provide sample_weight as the third element  of yourdataset, i.e. (inputs, targets, sample_weight). Received: x=%s, sample_weight=%s' % (x, sample_weight))
    if validation_split is not None and validation_split != 0.0:
        raise ValueError('`validation_split` argument is not supported when input `x` is a dataset or a dataset iterator. Received: x=%s, validation_split=%f' % (x, validation_split))