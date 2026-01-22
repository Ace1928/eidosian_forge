import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _EnforceShapeInvariant(merge_var, next_var):
    """Check if the shapes of the loops variables are invariants.

  Args:
    merge_var: The tensor representing the initial values of the loop
      variables.
    next_var: The tensor representing the values of the loop variables
      after one loop iteration.

  Raises:
    ValueError: If any tensor in `merge_var` has a more specific shape than
      its corresponding tensor in `next_var`.
  """
    if isinstance(merge_var, tensor_lib.Tensor):
        m_shape = merge_var.get_shape()
        n_shape = next_var.get_shape()
        if not _ShapeLessThanOrEqual(n_shape, m_shape):
            enter = merge_var.op.inputs[0].op
            assert util.IsLoopEnter(enter)
            input_t = enter.inputs[0]
            raise ValueError("Input tensor '%s' enters the loop with shape %s, but has shape %s after one iteration. To allow the shape to vary across iterations, use the `shape_invariants` argument of tf.while_loop to specify a less-specific shape." % (input_t.name, input_t.shape, n_shape))
    else:
        raise TypeError(f"'merge_var' must be a Tensor. Received: {type(merge_var)}.")