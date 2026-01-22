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
class CSRSparseMatrix(SparseMatrix):
    """(Optionally batched) CSR Sparse Matrix."""

    def __init__(self, value, indices=None, name=None):
        """Construct a CSRSparseMatrix from a dense matrix or SparseTensor.

    Args:
      value: A dense `2D` or `3D` Tensor or `SparseTensor`.
      indices: The nonzero indices of `value`
        (if `value` is not a `SparseTensor`).
      name: Optional op name.

    Raises:
      ValueError: if `value` is a `SparseTensor` and `indices` is not `None`.
    """
        del name
        super(CSRSparseMatrix, self).__init__()
        if isinstance(value, sparse_tensor.SparseTensor):
            if indices is not None:
                raise ValueError('indices must be None if value is a SparseTensor.')
            self._dtype = value.dtype
            self._csr_matrix = sm_ops.sparse_tensor_to_csr_sparse_matrix(indices=value.indices, values=value.values, dense_shape=value.dense_shape)
        else:
            value = ops.convert_to_tensor(value)
            self._dtype = value.dtype
            if indices is not None:
                indices = ops.convert_to_tensor(indices, dtype=dtypes.int64)
            else:
                indices = array_ops.stop_gradient(array_ops.where(value))
            self._csr_matrix = sm_ops.dense_to_csr_sparse_matrix(value, indices)
        if self._eager_mode:
            self._csr_matrix._handle_data = _make_handle_data(value)

    @property
    def _matrix(self):
        return self._csr_matrix

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

    def to_dense(self):
        return sm_ops.csr_sparse_matrix_to_dense(self._matrix, type=self.dtype)

    def to_sparse_tensor(self):
        r = sm_ops.csr_sparse_matrix_to_sparse_tensor(self._matrix, type=self.dtype)
        return sparse_tensor.SparseTensor(indices=r.indices, values=r.values, dense_shape=r.dense_shape)