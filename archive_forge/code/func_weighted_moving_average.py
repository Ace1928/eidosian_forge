from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def weighted_moving_average(value, decay, weight, truediv=True, collections=None, name=None):
    """Compute the weighted moving average of `value`.

  Conceptually, the weighted moving average is:
    `moving_average(value * weight) / moving_average(weight)`,
  where a moving average updates by the rule
    `new_value = decay * old_value + (1 - decay) * update`
  Internally, this Op keeps moving average variables of both `value * weight`
  and `weight`.

  Args:
    value: A numeric `Tensor`.
    decay: A float `Tensor` or float value. The moving average decay.
    weight:  `Tensor` that keeps the current value of a weight. Shape should be
      able to multiply `value`.
    truediv:  Boolean, if `True`, dividing by `moving_average(weight)` is
      floating point division.  If `False`, use division implied by dtypes.
    collections:  List of graph collections keys to add the internal variables
      `value * weight` and `weight` to. Defaults to
      `[GraphKeys.GLOBAL_VARIABLES]`.
    name: Optional name of the returned operation. Defaults to
      "WeightedMovingAvg".

  Returns:
    An Operation that updates and returns the weighted moving average.
  """
    if collections is None:
        collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    with variable_scope.variable_scope(name, 'WeightedMovingAvg', [value, weight, decay]) as scope:
        value_x_weight_var = variable_scope.get_variable('value_x_weight', shape=value.get_shape(), dtype=value.dtype, initializer=init_ops.zeros_initializer(), trainable=False, collections=collections)
        weight_var = variable_scope.get_variable('weight', shape=weight.get_shape(), dtype=weight.dtype, initializer=init_ops.zeros_initializer(), trainable=False, collections=collections)
        numerator = assign_moving_average(value_x_weight_var, value * weight, decay, zero_debias=False)
        denominator = assign_moving_average(weight_var, weight, decay, zero_debias=False)
        if truediv:
            return math_ops.truediv(numerator, denominator, name=scope.name)
        else:
            return math_ops.divide(numerator, denominator, name=scope.name)