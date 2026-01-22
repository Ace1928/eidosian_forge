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
class DistributedDatasetsFromFunctionSpec(DistributedDatasetAndIteratorSpec):
    """Type specification for `DistributedDatasetsFromFunction."""

    @property
    def value_type(self):
        return DistributedDatasetsFromFunction

    @property
    def _component_specs(self):
        specs = []
        worker_device_pairs = self._input_workers._worker_device_pairs
        for i, _ in enumerate(worker_device_pairs):
            element_spec = nest.map_structure(functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
            specs.append(dataset_ops.DatasetSpec(element_spec))
        return specs

    def _to_components(self, value):
        return value._datasets

    def _from_components(self, components):
        return DistributedDatasetsFromFunction(input_workers=self._input_workers, strategy=self._strategy, components=components, element_spec=self._element_spec, options=self._options)

    @staticmethod
    def from_value(value):
        return DistributedDatasetsFromFunctionSpec(input_workers=value._input_workers, element_spec=value._element_spec, strategy=value._strategy, options=value._options)