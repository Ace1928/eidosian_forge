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
class _SingleWorkerDatasetIteratorBase(object):
    """Iterator for a single `tf.data.Dataset`."""

    def __init__(self, dataset, worker, devices, options=None):
        """Create iterator for the `dataset` to fetch data to worker's `devices` .

    A `MultiDeviceIterator`  or `OwnedMultiDeviceIterator` is used to prefetch
    input to the devices on the given worker.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
      options: options.
    """
        self._dataset = dataset
        self._worker = worker
        self._devices = devices
        self._element_spec = dataset.element_spec
        self._options = options
        self._make_iterator()

    def _make_iterator(self):
        raise NotImplementedError('must be implemented in descendants')

    def _format_data_list_with_options(self, data_list):
        """Change the data in to a list type if required.

    The OwnedMultiDeviceIterator returns the list data type,
    while the PER_REPLICA iterator (when used with prefetch disabled)
    returns without the enclosed list. This is to fix the inconsistency.
    Args:
      data_list: data_list
    Returns:
      list
    """
        if self._options and self._options.experimental_replication_mode == InputReplicationMode.PER_REPLICA and (not self._options.experimental_fetch_to_device):
            return [data_list]
        else:
            return data_list

    def get_next(self, device, name=None):
        """Get next element for the given device."""
        del name
        with ops.device(self._worker):
            if _should_use_multi_device_iterator(self._options):
                return self._iterator.get_next(device)
            else:
                return self._iterator.get_next()

    def get_next_as_list(self, name=None):
        """Get next element from the underlying iterator.

    Runs the iterator get_next() within a device scope. Since this doesn't use
    get_next_as_optional(), it is considerably faster than get_next_as_list(),
    but it raises EOFError if any of the device doesn't get any data.

    Args:
      name: not used.

    Returns:
      A list consisting of the next data from each device.
    """
        del name
        with ops.device(self._worker):
            return self._format_data_list_with_options(self._iterator.get_next())

    def get_next_as_optional_list(self):
        with ops.device(self._worker):
            return self._format_data_list_with_options(self._iterator.get_next_as_optional())