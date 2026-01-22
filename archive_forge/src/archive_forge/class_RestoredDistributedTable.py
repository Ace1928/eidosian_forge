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
class RestoredDistributedTable(DistributedTable):
    """A restored and distributed StaticHashTable for ParameterServerStrategy."""

    def __init__(self, strategy, wrapped_creator):
        self._has_resource_functions = threading.Condition()
        super().__init__(strategy, wrapped_creator)

    def resource_handle_call_time_value(self):
        """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

        def closure():
            dispatch_context = coordinator_context.get_current_dispatch_context()
            if dispatch_context:
                local_resource_restore_context = get_current_local_resource_restore_context()
                if local_resource_restore_context:
                    remote_value = local_resource_restore_context.instance.resource_handle
                else:
                    remote_value = self._distributed_table._values[dispatch_context.worker_index]
                ret = dispatch_context.maybe_get_remote_value(remote_value)
                return ret
            else:
                return self._coordinator_instance.resource_handle
        return (closure, tensor.TensorSpec(shape=(), dtype=dtypes.resource))

    def __setattr__(self, name, value):
        if name in TRACKABLE_RESOURCE_METHODS:
            if not hasattr(self, '_restored_function'):
                self._restored_function = {}
            self._restored_function[name] = value
            if all((method in self._restored_function for method in TRACKABLE_RESOURCE_METHODS)):
                with self._has_resource_functions:
                    self._has_resource_functions.notify_all()
            return self._coordinator_instance.__setattr__(name, value)
        else:
            return super(RestoredDistributedTable, self).__setattr__(name, value)

    def _create_resource(self):
        """A function that creates a resource handle for a table on coordinator."""
        return self._coordinator_instance._create_resource()

    def _initialize(self):
        """A function that initializes the resource."""
        return self._coordinator_instance._initialize()

    def _destroy_resource(self):
        """A function that destroys the resource."""
        return self._coordinator_instance._destroy_resource()

    def _maybe_build_distributed_table(self):
        """Create table objects and resources on each worker if hasn't been created."""
        with self._distributed_table_creation_lock:
            if not self._distributed_table:

                def create_copy():
                    new_table = self._wrapped_creator()
                    with self._has_resource_functions:
                        while not hasattr(self, '_restored_function') or any((method not in self._restored_function for method in TRACKABLE_RESOURCE_METHODS)):
                            self._has_resource_functions.wait()
                    if hasattr(self, '_restored_function'):
                        with with_local_resource_restore_context(new_table):
                            for name, tf_function in self._restored_function.items():
                                setattr(new_table, name, tf_function)
                            init_op = new_table._initialize()
                            if not context.executing_eagerly():
                                ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
                    ret = new_table.resource_handle
                    return ret
                self._distributed_table = self._coordinator._create_per_worker_resources(create_copy)