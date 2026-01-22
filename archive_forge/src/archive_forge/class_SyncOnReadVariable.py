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
class SyncOnReadVariable(DistributedVariable):
    """Holds a map from replica to variables whose values are reduced on save."""

    def _update_replica(self, update_fn, value, **kwargs):
        return update_fn(self._get_on_device_or_primary(), value, **kwargs)

    def _get(self):
        """Returns the value of SyncOnReadVariable based on surrounding context.

    If called under a non-default replica-context, returns the corresponding
    variable on that replica.
    If called under default replica-context or cross-replica context, returns
    the synced value.
    """
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            return super(SyncOnReadVariable, self)._get()

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        if values_util.is_saving_non_distributed():
            return self._primary.assign_sub(value, use_locking, name, read_value)
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                values_util.mark_as_unsaveable()
                return values_util.on_read_assign_sub_cross_replica(self, value, read_value=read_value)
            else:
                return super(SyncOnReadVariable, self).assign_sub(value, use_locking, name, read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        if values_util.is_saving_non_distributed():
            return self._primary.assign_add(value, use_locking, name, read_value)
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                values_util.mark_as_unsaveable()
                return values_util.on_read_assign_add_cross_replica(self, value, read_value=read_value)
            else:
                return super(SyncOnReadVariable, self).assign_add(value, use_locking, name, read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if values_util.is_saving_non_distributed():
            return self._primary.assign(value, use_locking, name, read_value)
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                values_util.mark_as_unsaveable()
                return values_util.on_read_assign_cross_replica(self, value, read_value=read_value)
            else:
                return super(SyncOnReadVariable, self).assign(value, use_locking, name, read_value)

    def _scatter_not_implemented(self, method):
        raise NotImplementedError(f"Variables with `synchronization=ON_READ` doesn't support `{method}`")

    def scatter_sub(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_sub(*args, **kwargs)
        self._scatter_not_implemented('scatter_sub')

    def scatter_add(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_add(*args, **kwargs)
        self._scatter_not_implemented('scatter_add')

    def scatter_mul(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_mul(*args, **kwargs)
        self._scatter_not_implemented('scatter_mul')

    def scatter_div(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_div(*args, **kwargs)
        self._scatter_not_implemented('scatter_div')

    def scatter_min(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_min(*args, **kwargs)
        self._scatter_not_implemented('scatter_min')

    def scatter_max(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_max(*args, **kwargs)
        self._scatter_not_implemented('scatter_max')

    def scatter_update(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_update(*args, **kwargs)
        self._scatter_not_implemented('scatter_update')

    def value(self):
        if distribute_lib.in_variable_sync_on_read_context():
            raise NotImplementedError('call `variable.value()` inside variable_sync_on_read_context is not supported')
        if values_util.is_saving_non_distributed():
            return self._primary.value()
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
                    return self._get_replica(0).value()
                return self._get_cross_replica()
            else:
                return self._get_on_device_or_primary().value()

    def read_value(self):
        if distribute_lib.in_variable_sync_on_read_context():
            raise NotImplementedError('call `variable.read_value()` inside variable_sync_on_read_context is not supported')
        return super().read_value()

    def _get_cross_replica(self):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
            return self._get_replica(0)
        if self._aggregation == vs.VariableAggregation.SUM:
            values_util.mark_as_unsaveable()
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            return self._distribute_strategy.reduce(reduce_util.ReduceOp.from_variable_aggregation(self._aggregation), self, axis=None)

    def _as_graph_element(self):
        if values_util.is_saving_non_distributed():
            return self._primary._as_graph_element()
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            if distribute_lib.in_cross_replica_context():
                return ops.convert_to_tensor(self._get_cross_replica())
        return self._get()._as_graph_element()

    def _gather_saveables_for_checkpoint(self):
        """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    `SyncOnReadVariable`s.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

        def _saveable_factory(name=self._common_name):
            return _SyncOnReadSaveable(self, name)
        return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        """Converts a SyncOnReadVariable to a tensor."""
        if values_util.is_saving_non_distributed():
            return ops.convert_to_tensor(self._primary, dtype=dtype, name=name, as_ref=as_ref)
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            replica_context = distribute_lib.get_replica_context()
            if replica_context is not None and distribute_lib.in_variable_sync_on_read_context():
                if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
                    return ops.convert_to_tensor(self._get_replica(0), dtype=dtype, name=name, as_ref=as_ref)
                if self._aggregation == vs.VariableAggregation.SUM:
                    values_util.mark_as_unsaveable()
                reduced = replica_context.strategy.extended._replica_ctx_all_reduce(reduce_util.ReduceOp.from_variable_aggregation(self._aggregation), self._get().read_value())
                return ops.convert_to_tensor(reduced, dtype=dtype, name=name, as_ref=as_ref)
            return ops.convert_to_tensor(self._get(), dtype=dtype, name=name, as_ref=as_ref)