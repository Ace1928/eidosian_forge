import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
@ops.RegisterGradient('TopK')
@ops.RegisterGradient('TopKV2')
def _TopKGrad(op, grad, _):
    """Return the gradients for TopK.

  Args:
    op: The TopKOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the TopKOp.

  Returns:
    A list of two tensors, the first being the gradient w.r.t to the input and
    TopK, and the second being the gradient w.r.t. to the indices (all zero).
  """
    in_shape = array_ops.shape(op.inputs[0])
    ind_shape = array_ops.shape(op.outputs[1])
    ind_lastdim = array_ops.gather(math_ops.cast(ind_shape, dtypes.int64), array_ops.size(ind_shape) - 1)
    ind_2d = array_ops.reshape(op.outputs[1], array_ops_stack.stack([-1, ind_lastdim]))
    in_lastdim = array_ops.gather(math_ops.cast(in_shape, dtypes.int64), array_ops.size(in_shape) - 1)
    outerdim = array_ops.shape(ind_2d)[0]
    ind = array_ops.reshape(ind_2d + math_ops.cast(array_ops.expand_dims(math_ops.range(0, math_ops.cast(outerdim, dtypes.int64) * in_lastdim, in_lastdim), -1), dtypes.int32), [-1])
    return [array_ops.reshape(array_ops.scatter_nd(array_ops.expand_dims(ind, -1), array_ops.reshape(grad, [-1]), [math_ops.reduce_prod(in_shape)]), in_shape), array_ops.zeros([], dtype=dtypes.int32)]