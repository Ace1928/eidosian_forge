from enum import Enum
import functools
import weakref
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_control_flow_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def update_state_wrapper(update_state_fn):
    """Decorator to wrap metric `update_state()` with `add_update()`.

  Args:
    update_state_fn: function that accumulates metric statistics.

  Returns:
    Decorated function that wraps `update_state_fn()` with `add_update()`.
  """

    def decorated(metric_obj, *args, **kwargs):
        """Decorated function with `add_update()`."""
        strategy = distribute_lib.get_strategy()
        for weight in metric_obj.weights:
            if backend.is_tpu_strategy(strategy) and (not strategy.extended.variable_created_in_scope(weight)) and (not distribute_lib.in_cross_replica_context()):
                raise ValueError('Trying to run metric.update_state in replica context when the metric was not created in TPUStrategy scope. Make sure the keras Metric is created in TPUstrategy scope. ')
        with tf_utils.graph_context_for_symbolic_tensors(*args, **kwargs):
            update_op = update_state_fn(*args, **kwargs)
        if update_op is not None:
            metric_obj.add_update(update_op)
        return update_op
    return tf_decorator.make_decorator(update_state_fn, decorated)