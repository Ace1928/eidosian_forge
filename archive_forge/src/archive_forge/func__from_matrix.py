import abc
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *
def _from_matrix(self, matrix, handle_data=None):
    assert isinstance(matrix, tensor_lib.Tensor) and matrix.dtype == dtypes.variant
    ret = type(self).__new__(type(self))
    ret._dtype = self._dtype
    if self._eager_mode:
        if matrix._handle_data is None:
            matrix._handle_data = handle_data
        assert matrix._handle_data is not None
    ret._csr_matrix = matrix
    return ret