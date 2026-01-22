import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
def SmartBroadcastGradientArgs(x, y, grad):
    """Optimized version of `broadcast_gradient_args` that caches results.

  This implementation avoids creating `broadcast_gradient_args` ops in the case
  that the input shapes are fully defined, and provides hints to the calling
  code that can be used to avoid creating reduction and reshaping ops.

  Args:
    x: The left input tensor to a broadcasting binary op.
    y: The right input tensor to a broadcasting binary op.
    grad: The incoming gradient tensor for a broadcasting binary op.

  Returns:
    A pair of tuples, containing:
      * A 3-tuple of broadcast information for x, containing:
        * The shape of x (as a tuple or Tensor).
        * The reduction indices for x (as a tuple or Tensor).
        * A boolean, which if True, indicates that x's shape differs from grad's
          shape (and so x's gradient must be reduced and/or reshaped).
      * A 3-tuple of broadcast information for y, containing the respective
        details for y.
  """
    if context.executing_eagerly() or not (isinstance(x, tensor.Tensor) and isinstance(y, tensor.Tensor) and isinstance(grad, tensor.Tensor)):
        sx = array_ops.shape(x)
        sy = array_ops.shape(y)
        rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
        return ((sx, rx, True), (sy, ry, True))
    x_shape_tuple = x._shape_tuple()
    y_shape_tuple = y._shape_tuple()
    grad_shape_tuple = grad._shape_tuple()
    if x_shape_tuple is None or None in x_shape_tuple or y_shape_tuple is None or (None in y_shape_tuple):
        sx = array_ops.shape_internal(x, optimize=False)
        sy = array_ops.shape_internal(y, optimize=False)
        rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
        return ((sx, rx, True), (sy, ry, True))
    x_needs_reduction = x_shape_tuple != grad_shape_tuple
    y_needs_reduction = y_shape_tuple != grad_shape_tuple
    g = ops.get_default_graph()
    try:
        rx, ry = g._bcast_grad_args_cache[x_shape_tuple, y_shape_tuple]
        return ((x_shape_tuple, rx, x_needs_reduction), (y_shape_tuple, ry, y_needs_reduction))
    except KeyError:
        rx, ry = array_ops.broadcast_gradient_args(x_shape_tuple, y_shape_tuple)
        rx_value = tuple(tensor_util.try_evaluate_constant(rx))
        assert rx_value is not None
        ry_value = tuple(tensor_util.try_evaluate_constant(ry))
        assert ry_value is not None
        g._bcast_grad_args_cache[x_shape_tuple, y_shape_tuple] = (rx_value, ry_value)
        return ((x_shape_tuple, rx_value, x_needs_reduction), (y_shape_tuple, ry_value, y_needs_reduction))