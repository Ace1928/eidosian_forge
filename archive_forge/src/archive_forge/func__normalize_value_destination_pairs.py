import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _normalize_value_destination_pairs(value_destination_pairs):
    """Converts each tensor into a PerReplica object in the input list."""
    result = []
    value_destination_pairs = list(value_destination_pairs)
    if not isinstance(value_destination_pairs, (list, tuple)):
        raise ValueError('`value_destination_pairs` should be a list or tuple')
    for pair in value_destination_pairs:
        if not isinstance(pair, tuple):
            raise ValueError('Each element of `value_destination_pairs` should be a tuple.')
        if len(pair) != 2:
            raise ValueError('Each element of `value_destination_pairs` should be a tuple of size 2.')
        per_replica = _make_tensor_into_per_replica(pair[0])
        result.append((per_replica, pair[1]))
    return result