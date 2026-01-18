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
def validate_input_types(inp, orig_inp, allow_dict=True, field_name='inputs'):
    """Helper function to validate either inputs or targets."""
    if isinstance(inp, (list, tuple)):
        if not all((isinstance(v, np.ndarray) or tensor_util.is_tf_type(v) for v in inp)):
            raise ValueError('Please provide as model inputs either a single array or a list of arrays. You passed: {}={}'.format(field_name, str(orig_inp)))
    elif isinstance(inp, dict):
        if not allow_dict:
            raise ValueError('You cannot pass a dictionary as model {}.'.format(field_name))
    elif not isinstance(inp, np.ndarray) and (not tensor_util.is_tf_type(inp)):
        raise ValueError('Please provide as model inputs either a single array or a list of arrays. You passed: {}={}'.format(field_name, orig_inp))