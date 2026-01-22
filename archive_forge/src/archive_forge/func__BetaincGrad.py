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
@ops.RegisterGradient('Betainc')
def _BetaincGrad(op, grad):
    """Returns gradient of betainc(a, b, x) with respect to x."""
    a, b, x = op.inputs
    sa = array_ops.shape(a)
    sx = array_ops.shape(x)
    _, rx = gen_array_ops.broadcast_gradient_args(sa, sx)
    log_beta = gen_math_ops.lgamma(a) + gen_math_ops.lgamma(b) - gen_math_ops.lgamma(a + b)
    partial_x = math_ops.exp(math_ops.xlog1py(b - 1, -x) + math_ops.xlogy(a - 1, x) - log_beta)
    return (None, None, array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))