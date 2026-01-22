import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
class DistributedTable(lookup_ops.StaticHashTable):
    """A distributed StaticHashTable for ParameterServerStrategy.

  An instance of DistributedTable has copies of a StaticHashTable and its
  resource handle on the coordinator of each worker, created at the
  DistributedTable instance initialization time with initializers on each
  worker. Users can call methods on a DistributedTable as if it were a
  StaticHashTable, which leads to execution with the resource local to the
  consumer worker (or the coordinator, if calling from the coordinator). This
  implementation relies on the fact that the methods of StaticHashTable are
  queried with the resource handle (instead of the python object).

  Currently, at saving time, a DistributedTable is saved as a StaticHashTable on
  the coordinator, and restoring a DistributedTable from SavedModel is not
  supported.
  """

    def __init__(self, strategy, wrapped_creator):
        distribute_lib.distribution_strategy_input_api_counter.get_cell(self.__class__.__name__, 'PSSDistributedLookupTable').increase_by(1)
        self._coordinator_instance = wrapped_creator()
        self._wrapped_creator = wrapped_creator
        self._coordinator = strategy._cluster_coordinator
        self._distributed_table = None
        self._distributed_table_creation_lock = threading.Lock()
        if not save_context.in_save_context():
            self._maybe_build_distributed_table()

    def __getattr__(self, attr):
        if attr == '_coordinator_instance':
            raise AttributeError()
        if attr in self._coordinator_instance.__dict__:
            attr_value = self._coordinator_instance.__dict__[attr]
            if callable(attr_value):

                def wrapper(*args, **kwargs):
                    return attr_value(self, *args, **kwargs)
                return wrapper
            elif isinstance(attr_value, property):
                return attr_value
            else:
                return getattr(self._coordinator_instance, attr)
        else:
            return getattr(self._coordinator_instance, attr)

    def resource_handle_call_time_value(self):
        """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

        def closure():
            dispatch_context = coordinator_context.get_current_dispatch_context()
            if dispatch_context:
                remote_value = self._distributed_table._values[dispatch_context.worker_index]
                ret = dispatch_context.maybe_get_remote_value(remote_value)
                return ret
            else:
                return self._coordinator_instance.resource_handle
        return (closure, tensor.TensorSpec([], dtype=dtypes.resource))

    def _maybe_build_distributed_table(self):
        """Create table objects and resources on each worker if hasn't been created."""
        with self._distributed_table_creation_lock:
            if not self._distributed_table:

                def create_copy():
                    new_table = self._wrapped_creator()
                    ret = new_table.resource_handle
                    return ret
                self._distributed_table = self._coordinator._create_per_worker_resources(create_copy)

    @property
    def resource_handle(self):
        if context.executing_eagerly() or save_context.in_save_context():
            return self._coordinator_instance.resource_handle
        else:
            self._maybe_build_distributed_table()
            closure, spec = self.resource_handle_call_time_value()
            return ops.get_default_graph().capture_call_time_value(closure, spec, default_value=self._coordinator_instance.resource_handle)

    @property
    def is_distributed_table(self):
        return True

    def __tf_experimental_restore_capture__(self, concrete_function, internal_capture):
        closure, spec = self.resource_handle_call_time_value()
        concrete_function.graph.replace_capture_with_deferred_capture(self._coordinator_instance.resource_handle, closure, spec, default_value=self._coordinator_instance.resource_handle, placeholder=internal_capture)
        return concrete_function.graph.deferred_external_captures[-1]