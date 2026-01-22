from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
def _check_row_col(self, row, col):
    """Static check of row and column."""
    for name, tensor in [['row', row], ['col', col]]:
        if tensor.shape.ndims is not None and tensor.shape.ndims < 1:
            raise ValueError('Argument {} must have at least 1 dimension.  Found: {}'.format(name, tensor))
    if row.shape[-1] is not None and col.shape[-1] is not None:
        if row.shape[-1] != col.shape[-1]:
            raise ValueError('Expected square matrix, got row and col with mismatched dimensions.')