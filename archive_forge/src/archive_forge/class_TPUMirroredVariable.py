from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
class TPUMirroredVariable(TPUVariableMixin, values.MirroredVariable):
    """Holds a map from replica to TPU variables whose values are kept in sync."""

    def _is_replicated_or_sharded_to_logical_cores(self):
        """Returns whether each of the underlying variables is replicated or sharded to logical cores.

    If True, the handles of the underlying variables are not available outside a
    TPU context.
    """
        return isinstance(self._primary, tpu_replicated_variable.TPUReplicatedVariable)

    @property
    def device(self):
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_util.enclosing_tpu_context() is None:
            return self._primary.device
        return super(TPUMirroredVariable, self).device

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        tpu_context = tpu_util.enclosing_tpu_context()
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_context is None:
            assign_sub_fn = lambda v, *a, **ka: v.assign_sub(*a, **ka)
            return self._update(update_fn=assign_sub_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        if tpu_context and self.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_sub(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        tpu_context = tpu_util.enclosing_tpu_context()
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_context is None:
            assign_add_fn = lambda v, *a, **ka: v.assign_add(*a, **ka)
            return self._update(update_fn=assign_add_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        if tpu_context and self.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_add(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        tpu_context = tpu_util.enclosing_tpu_context()
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_context is None:
            assign_fn = lambda v, *a, **ka: v.assign(*a, **ka)
            return self._update(update_fn=assign_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        if tpu_util.enclosing_tpu_context() and self.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def scatter_sub(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_sub(*args, **kwargs)
        raise NotImplementedError

    def scatter_add(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_add(*args, **kwargs)
        raise NotImplementedError

    def scatter_max(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_max(*args, **kwargs)
        raise NotImplementedError

    def scatter_min(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_min(*args, **kwargs)
        raise NotImplementedError

    def scatter_mul(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_mul(*args, **kwargs)
        raise NotImplementedError

    def scatter_div(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_div(*args, **kwargs)
        raise NotImplementedError

    def scatter_update(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_update(*args, **kwargs)
        raise NotImplementedError