import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
def _apply_assign_update(self, update_fn, value, use_locking=None, name=None, read_value=True):
    if ops.executing_eagerly_outside_functions():
        assign_op = update_fn(value, use_locking, name, False)
        if read_value:
            var = create_autocast_variable(self._variable)
            var._op = assign_op
            return var
        return assign_op
    assign_var = update_fn(value, use_locking, name, read_value)
    if read_value and resource_variable_ops.is_resource_variable(assign_var):
        return create_autocast_variable(assign_var)
    return assign_var