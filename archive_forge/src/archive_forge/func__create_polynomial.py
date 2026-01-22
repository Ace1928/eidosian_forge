import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def _create_polynomial(var, coeffs):
    """Compute n_th order polynomial via Horner's method."""
    coeffs = np.array(coeffs, var.dtype.as_numpy_dtype)
    if not coeffs.size:
        return array_ops.zeros_like(var)
    return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var