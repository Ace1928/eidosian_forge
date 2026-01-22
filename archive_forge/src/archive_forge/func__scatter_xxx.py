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
def _scatter_xxx(self, raw_scater_xxx_fn, op_name, var, sparse_delta, use_locking=False, name=None):
    scater_xxx_fn = tpu_util.make_raw_scatter_xxx_fn(raw_scater_xxx_fn)
    if tpu_util.enclosing_tpu_context():
        if self._aggregation != variable_scope.VariableAggregation.NONE:
            raise NotImplementedError(_scatter_error_msg.format(op_name=op_name, aggregation=self._aggregation))
        return scater_xxx_fn(var, sparse_delta=sparse_delta, use_locking=use_locking, name=name)
    else:
        return var._update(update_fn=scater_xxx_fn, value=sparse_delta, use_locking=use_locking, name=name)