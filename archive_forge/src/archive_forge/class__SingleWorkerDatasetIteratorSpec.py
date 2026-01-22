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
class _SingleWorkerDatasetIteratorSpec(type_spec.TypeSpec):
    """Type specification for `_SingleWorkerOwnedDatasetIterator`."""
    __slots__ = ['_worker', '_devices', '_element_spec', '_options', '_canonicalize_devices']

    def __init__(self, worker, devices, element_spec, options, canonicalize_devices=True):
        self._worker = worker
        if canonicalize_devices:
            self._devices = tuple((device_util.canonicalize(d) for d in devices))
        else:
            self._devices = tuple((device_util.canonicalize_without_job_and_task(d) for d in devices))
        self._element_spec = element_spec
        self._options = options if options is not None else distribute_lib.InputOptions()
        self._canonicalize_devices = canonicalize_devices

    @property
    def value_type(self):
        return _SingleWorkerOwnedDatasetIterator

    def _serialize(self):
        return (self._worker, self._devices, self._element_spec, self._options, self._canonicalize_devices)

    def _get_multi_device_iterator_spec(self, specs):
        device_scope = device_util.canonicalize(self._worker, device_util.current())
        host_device = device_util.get_host_for_device(device_scope)
        worker = host_device
        specs.append(multi_device_iterator_ops.MultiDeviceIteratorSpec(self._devices, worker, element_spec=self._element_spec))

    @property
    def _component_specs(self):
        specs = []
        if _should_use_multi_device_iterator(self._options):
            self._get_multi_device_iterator_spec(specs)
        else:
            specs.append(iterator_ops.IteratorSpec(element_spec=self._element_spec))
        return specs

    def _to_components(self, value):
        return [value._iterator]

    def _from_components(self, components):
        return _SingleWorkerOwnedDatasetIterator(dataset=None, worker=self._worker, devices=self._devices, components=components, element_spec=self._element_spec, options=self._options, canonicalize_devices=self._canonicalize_devices)

    @staticmethod
    def from_value(value):
        return _SingleWorkerDatasetIteratorSpec(value._worker, value._devices, value._element_spec, value._options, value._canonicalize_devices)