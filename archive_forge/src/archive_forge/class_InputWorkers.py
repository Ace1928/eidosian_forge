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
class InputWorkers(object):
    """A 1-to-many mapping from input worker devices to compute devices."""

    def __init__(self, worker_device_pairs, canonicalize_devices=True):
        """Initialize an `InputWorkers` object.

    Args:
      worker_device_pairs: A sequence of pairs: `(input device, a tuple of
        compute devices fed by that input device)`.
      canonicalize_devices: Whether to canonicalize devices for workers fully or
        partially. If False, it will partially canonicalize devices by removing
        job and task.
    """
        self._worker_device_pairs = worker_device_pairs
        self._input_worker_devices = tuple((d for d, _ in self._worker_device_pairs))
        self._canonicalize_devices = canonicalize_devices
        if canonicalize_devices:
            self._fed_devices = tuple((tuple((device_util.canonicalize(d) for d in f)) for _, f in self._worker_device_pairs))
        else:
            self._fed_devices = tuple((tuple((device_util.canonicalize_without_job_and_task(d) for d in f)) for _, f in self._worker_device_pairs))

    @property
    def num_workers(self):
        return len(self._input_worker_devices)

    @property
    def worker_devices(self):
        return self._input_worker_devices

    def compute_devices_for_worker(self, worker_index):
        return self._fed_devices[worker_index]

    def __repr__(self):
        devices = self.worker_devices
        debug_repr = ',\n'.join(('  %d %s: %s' % (i, devices[i], self._fed_devices[i]) for i in range(len(devices))))
        return '%s:{\n%s}' % (self.__class__.__name__, debug_repr)

    def serialize(self):
        return (self._worker_device_pairs, self._canonicalize_devices)

    def deserialize(self, serialized):
        return InputWorkers(serialized)