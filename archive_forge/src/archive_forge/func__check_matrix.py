from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _check_matrix(self, matrix):
    """Static check of the `matrix` argument."""
    allowed_dtypes = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]
    matrix = tensor_conversion.convert_to_tensor_v2_with_dispatch(matrix, name='matrix')
    dtype = matrix.dtype
    if dtype not in allowed_dtypes:
        raise TypeError(f'Argument `matrix` must have dtype in {allowed_dtypes}. Received: {dtype}.')
    if matrix.shape.ndims is not None and matrix.shape.ndims < 2:
        raise ValueError(f'Argument `matrix` must have at least 2 dimensions. Received: {matrix}.')