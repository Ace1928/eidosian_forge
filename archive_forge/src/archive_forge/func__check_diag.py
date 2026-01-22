from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _check_diag(self, diag):
    """Static check of diag."""
    if diag.shape.ndims is not None and diag.shape.ndims < 1:
        raise ValueError('Argument diag must have at least 1 dimension.  Found: %s' % diag)