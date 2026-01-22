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
class DistributedIterator(DistributedIteratorBase, composite_tensor.CompositeTensor):
    """Input Iterator for a distributed dataset."""

    def __init__(self, input_workers=None, iterators=None, strategy=None, components=None, element_spec=None, cardinality=cardinality_lib.UNKNOWN, enable_get_next_as_optional=False, options=None, replica_order=None):
        if input_workers is None:
            raise ValueError('`input_workers` should be provided.')
        error_message = 'Either `input_workers` or both `components` and `element_spec` need to be provided.'
        self._options = options
        if iterators is None:
            if components is None or element_spec is None:
                raise ValueError(error_message)
            self._element_spec = element_spec
            self._input_workers = input_workers
            self._iterators = components
            self._strategy = strategy
            self._cardinality = cardinality
            self._enable_get_next_as_optional = enable_get_next_as_optional
            self._replica_order = replica_order
        else:
            if components is not None and element_spec is not None:
                raise ValueError(error_message)
            super(DistributedIterator, self).__init__(input_workers, iterators, strategy, cardinality, enable_get_next_as_optional, replica_order)

    @property
    def element_spec(self):
        if self._enable_get_next_as_optional and self._strategy.extended._in_multi_worker_mode():
            return nest.map_structure(_rebatch_as_dynamic, self._element_spec, expand_composites=False)
        return self._element_spec

    @property
    def _type_spec(self):
        return DistributedIteratorSpec(self._input_workers, self._element_spec, self._strategy, self._options, self._cardinality, self._enable_get_next_as_optional, self._replica_order)