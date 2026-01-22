import copy
from typing import Optional
import weakref
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace
class OnWritePolicy(VariablePolicy):
    """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when the following `synchronization` and `aggregation`
  parameters are specified when creating a `tf.Variable` in `tf.distribute`
  scope and `synchronization` is equal to `tf.VariableSynchronization.ON_WRITE`
  or `tf.VariableSynchronization.AUTO`.
  """

    def _is_mirrored(self):
        return True

    def value(self, var):
        return var._get_on_device_or_primary().value()

    def _as_graph_element(self, var):
        return var._get_on_device_or_primary()._as_graph_element()

    def _get_cross_replica(self, var):
        return array_ops.identity(var._get_on_device_or_primary())

    def _update_replica(self, var, update_fn, value, **kwargs):
        if var.aggregation == variables_lib.VariableAggregation.NONE:
            return update_fn(var._get_on_device_or_primary(), value, **kwargs)
        return _on_write_update_replica(var, update_fn, value, **kwargs)

    def assign(self, var, value, use_locking=False, name=None, read_value=True):
        return values_util.on_write_assign(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, var, value, use_locking=False, name=None, read_value=True):
        return values_util.on_write_assign_add(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_sub(self, var, value, use_locking=False, name=None, read_value=True):
        return values_util.on_write_assign_sub(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
        return values_util.scatter_sub(var, sparse_delta, use_locking=use_locking, name=name)

    def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
        return values_util.scatter_add(var, sparse_delta, use_locking=use_locking, name=name)

    def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
        return values_util.scatter_mul(var, sparse_delta, use_locking=use_locking, name=name)

    def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
        return values_util.scatter_div(var, sparse_delta, use_locking=use_locking, name=name)

    def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
        if self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and self._aggregation != vs.VariableAggregation.NONE:
            raise NotImplementedError(values_util.scatter_error_msg.format(op_name='scatter_min', aggregation=self._aggregation))
        return values_util.scatter_min(var, sparse_delta, use_locking=use_locking, name=name)

    def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
        if self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and self._aggregation != vs.VariableAggregation.NONE:
            raise NotImplementedError(values_util.scatter_error_msg.format(op_name='scatter_max', aggregation=self._aggregation))
        return values_util.scatter_max(var, sparse_delta, use_locking=use_locking, name=name)

    def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
        if self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and self._aggregation != vs.VariableAggregation.NONE:
            raise NotImplementedError(values_util.scatter_error_msg.format(op_name='scatter_update', aggregation=self._aggregation))
        return values_util.scatter_update(var, sparse_delta, use_locking=use_locking, name=name)

    def get_saveable(self, var, primary_var, name):
        """Saveable ops for AUTO variables."""
        return values_util.get_on_write_saveable(var, primary_var, name)

    def get_restore_ops(self, var, tensor):
        return values_util.get_on_write_restore_ops(var, tensor)