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
def _all_reduce_per_replica_values(self, reduce_op, per_replica_values, options):
    """All reduce a list of per_replica_value."""
    values_by_device = [[] for _ in self._devices]
    num_devices = len(self._devices)
    for per_replica in per_replica_values:
        for i in range(num_devices):
            values_by_device[i].append(per_replica.values[i])
    if context.executing_eagerly():

        def thread_fn(device_id):
            with context.eager_mode():
                return self._all_reduce(reduce_op, values_by_device[device_id], device_id, options)
        with self._lock:
            pool = multiprocessing.pool.ThreadPool(len(self._devices))
            outputs_by_device = pool.map(thread_fn, list(range(num_devices)))
            pool.close()
    else:
        outputs_by_device = []
        with self._lock:
            for i in range(num_devices):
                outputs_by_device.append(self._all_reduce(reduce_op, values_by_device[i], i, options))
    result = []
    for values in zip(*outputs_by_device):
        result.append(distribute_utils.regroup(values, wrap_class=value_lib.Mirrored))
    return result