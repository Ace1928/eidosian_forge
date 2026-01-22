import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def _double_factorial(n):
    """The double factorial function for small Python integer `n`."""
    return np.prod(np.arange(n, 1, -2))