import abc
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def update_if_not_finite_grads():
    """Update assuming the gradients are nonfinite."""
    new_loss_scale = math_ops.maximum(self._current_loss_scale / self._multiplier, 1)
    return control_flow_ops.group(self._num_good_steps.assign(0), self._current_loss_scale.assign(new_loss_scale))