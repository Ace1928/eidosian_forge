import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('RandomGamma')
def _RandomGammaGrad(op, grad):
    """Returns the gradient of a Gamma sample w.r.t. alpha.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `RandomGamma` operation. We assume that the inputs to the operation
      are `shape` and `alpha` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A `Tensor` with derivatives `dloss / dalpha`.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
    shape = op.inputs[0]
    alpha = op.inputs[1]
    sample = op.outputs[0]
    with ops.control_dependencies([grad]):
        num_sample_dimensions = array_ops.shape(shape)[0]
        alpha_broadcastable = add_leading_unit_dimensions(alpha, num_sample_dimensions)
        partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)
        return (None, math_ops.reduce_sum(grad * partial_a, axis=math_ops.range(num_sample_dimensions)))