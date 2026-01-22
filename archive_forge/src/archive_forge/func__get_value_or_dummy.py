import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _get_value_or_dummy(input_workers, optional_list, produce_dummy):
    """Returns the value of the optionals or dummy values.

  Args:
    input_workers: the `InputWorkers`.
    optional_list: a list of lists `tf.experimental.Optional`. The values from
      each compute device grouped by the input device.
    produce_dummy: a bool. Whether to produce dummy tensors when the optional
      doesn't have a value.

  Returns:
    A flatten list of Tensors.

  """
    value_list = []
    for i, worker in enumerate(input_workers.worker_devices):
        with ops.device(worker):
            devices = input_workers.compute_devices_for_worker(i)
            for j, device in enumerate(devices):
                with ops.device(device):
                    if produce_dummy:
                        value_list.append(tf_cond.cond(optional_list[i][j].has_value(), lambda: optional_list[i][j].get_value(), lambda: _dummy_tensor_fn(optional_list[i][j].element_spec), strict=True))
                    else:
                        value_list.append(optional_list[i][j].get_value())
    return value_list