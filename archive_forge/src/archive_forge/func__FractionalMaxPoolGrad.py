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
@ops.RegisterGradient('FractionalMaxPool')
def _FractionalMaxPoolGrad(op, grad_0, unused_grad_1, unused_grad_2):
    """Returns gradient for FractionalMaxPool.

  Since FractionalMaxPool has three outputs, there are three gradients passed in
  for each of the outputs. Only the first one is useful, the other two gradients
  are empty.

  Args:
    op: The FractionalMaxPoolOp.
    grad_0: Gradient with respect to op.outputs[0]
    unused_grad_1: Gradient with respect to op.outputs[1]/row_seq. It is empty.
    unused_grad_2: Gradient with respect to op.outputs[2]/col_seq. It is empty.

  Returns:
    Input backprop for FractionalMaxPool op.
  """
    return gen_nn_ops.fractional_max_pool_grad(op.inputs[0], op.outputs[0], grad_0, op.outputs[1], op.outputs[2], op.get_attr('overlapping'))