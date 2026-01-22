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
@ops.RegisterGradient('BiasAdd')
def _BiasAddGrad(op, received_grad):
    """Return the gradients for the 2 inputs of bias_op.

  The first input of unused_bias_op is the tensor t, and its gradient is
  just the gradient the unused_bias_op received.

  The second input of unused_bias_op is the bias vector which has one fewer
  dimension than "received_grad" (the batch dimension.)  Its gradient is the
  received gradient Summed on the batch dimension, which is the first dimension.

  Args:
    op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.

  Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
  """
    try:
        data_format = op.get_attr('data_format')
    except ValueError:
        data_format = None
    return (received_grad, gen_nn_ops.bias_add_grad(out_backprop=received_grad, data_format=data_format))