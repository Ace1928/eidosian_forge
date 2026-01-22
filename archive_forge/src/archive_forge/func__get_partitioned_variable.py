import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _get_partitioned_variable(self, var_store, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
    """Gets an existing variable with this name or create a new one."""
    if initializer is None:
        initializer = self._initializer
    if regularizer is None:
        regularizer = self._regularizer
    if constraint is None:
        constraint = self._constraint
    if caching_device is None:
        caching_device = self._caching_device
    if partitioner is None:
        partitioner = self._partitioner
    if dtype is None:
        dtype = self._dtype
    if use_resource is None:
        use_resource = self._use_resource
    if self._custom_getter is not None:
        raise ValueError("Private access to _get_partitioned_variable is not allowed when a custom getter is set.  Current custom getter: %s.  It is likely that you're using create_partitioned_variables.  If so, consider instead using get_variable with a non-empty partitioner parameter instead." % self._custom_getter)
    if partitioner is None:
        raise ValueError('No partitioner was specified')
    full_name_list = []
    if self.name:
        full_name_list.append(self.name)
    if name:
        full_name_list.append(name)
    full_name = '/'.join(full_name_list)
    with ops.name_scope(None, skip_on_eager=False):
        return var_store._get_partitioned_variable(full_name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=self.reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)