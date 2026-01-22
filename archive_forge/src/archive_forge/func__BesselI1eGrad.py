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
@ops.RegisterGradient('BesselI1e')
def _BesselI1eGrad(op, grad):
    """Compute gradient of bessel_i1e(x) with respect to its argument."""
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        dy_dx = array_ops.where_v2(math_ops.equal(x, 0.0), math_ops.cast(0.5, x.dtype), special_math_ops.bessel_i0e(x) - y * (math_ops.sign(x) + math_ops.reciprocal(x)))
        return grad * dy_dx