import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
Returns the gradient of a TruncatedNormal sample w.r.t. parameters.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `StatelessParameterizedTruncatedNormal` operation. We assume that the
      inputs to the operation are `shape`, `seed`, `mean`, `stddev`, `minval`,
      and `maxval` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A list of `Tensor` with derivates with respect to each parameter.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  