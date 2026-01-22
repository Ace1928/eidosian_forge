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
def _apply_update(self, update_fn, *args, **kwargs):
    update_var = update_fn(*args, **kwargs)
    if ops.executing_eagerly_outside_functions():
        return self
    if resource_variable_ops.is_resource_variable(update_var):
        return create_autocast_variable(update_var)
    return update_var