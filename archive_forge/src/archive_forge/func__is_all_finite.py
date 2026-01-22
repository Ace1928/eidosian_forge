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
def _is_all_finite(grads):
    """Returns a scalar boolean tensor indicating if all gradients are finite."""

    def raw_values(g):
        return g.values if isinstance(g, indexed_slices.IndexedSlices) else g
    is_finite_per_grad = [math_ops.reduce_all(math_ops.is_finite(raw_values(g))) for g in grads if g is not None]
    return math_ops.reduce_all(is_finite_per_grad)