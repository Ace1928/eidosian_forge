import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_arg_values
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _block_orth(self, p1, p2, p3):
    """Construct a 3 x 3 kernel.

    Used to construct orthgonal kernel.

    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.
      p3: A symmetric projection matrix.

    Returns:
      A 2 x 2 x 2 kernel.
    Raises:
      ValueError: If the dimensions of p1, p2 and p3 are different.
    """
    p1_shape = p1.shape.as_list()
    if p1_shape != p2.shape.as_list() or p1_shape != p3.shape.as_list():
        raise ValueError(f'The dimension of the matrices must be the same. Received p1.shape={p1.shape}, p2.shape={p2.shape} and p3.shape={p3.shape}.')
    n = p1_shape[0]
    eye = linalg_ops_impl.eye(n, dtype=self.dtype)
    kernel2x2x2 = {}

    def matmul(p1, p2, p3):
        return math_ops.matmul(math_ops.matmul(p1, p2), p3)

    def cast(i, p):
        """Return p or (1-p)."""
        return i * p + (1 - i) * (eye - p)
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                kernel2x2x2[i, j, k] = matmul(cast(i, p1), cast(j, p2), cast(k, p3))
    return kernel2x2x2