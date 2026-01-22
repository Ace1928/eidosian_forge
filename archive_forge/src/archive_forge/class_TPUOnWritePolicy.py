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
class TPUOnWritePolicy(values.OnWritePolicy):
    """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.AUTO` or `tf.VariableSynchronization.ON_WRITE`.
  """

    def assign_sub(self, var, value, use_locking=False, name=None, read_value=True):
        if tpu_util.enclosing_tpu_context() and var.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(var, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_sub(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, var, value, use_locking=False, name=None, read_value=True):
        if tpu_util.enclosing_tpu_context() and var.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(var, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_add(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, var, value, use_locking=False, name=None, read_value=True):
        if tpu_util.enclosing_tpu_context() and var.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(var, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def _scatter_xxx(self, raw_scater_xxx_fn, op_name, var, sparse_delta, use_locking=False, name=None):
        scater_xxx_fn = tpu_util.make_raw_scatter_xxx_fn(raw_scater_xxx_fn)
        if tpu_util.enclosing_tpu_context():
            if self._aggregation != variable_scope.VariableAggregation.NONE:
                raise NotImplementedError(_scatter_error_msg.format(op_name=op_name, aggregation=self._aggregation))
            return scater_xxx_fn(var, sparse_delta=sparse_delta, use_locking=use_locking, name=name)
        else:
            return var._update(update_fn=scater_xxx_fn, value=sparse_delta, use_locking=use_locking, name=name)

    def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_sub, 'scatter_sub', var, sparse_delta, use_locking, name)

    def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_add, 'scatter_add', var, sparse_delta, use_locking, name)

    def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_max, 'scatter_max', var, sparse_delta, use_locking, name)

    def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_min, 'scatter_min', var, sparse_delta, use_locking, name)

    def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_mul, 'scatter_mul', var, sparse_delta, use_locking, name)

    def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_div, 'scatter_div', var, sparse_delta, use_locking, name)

    def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_update, 'scatter_update', var, sparse_delta, use_locking, name)