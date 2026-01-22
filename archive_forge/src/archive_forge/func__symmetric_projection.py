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
def _symmetric_projection(self, n):
    """Compute a n x n symmetric projection matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = self._orthogonal_matrix(n)
    mask = math_ops.cast(random_ops.random_normal([n], seed=self.seed) > 0, self.dtype)
    if self.seed:
        self.seed += 1
    c = math_ops.multiply(q, mask)
    return math_ops.matmul(c, array_ops.matrix_transpose(c))