import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('sparse_tensor_type', [pa.SparseCSRMatrix, pa.SparseCSCMatrix, pa.SparseCOOTensor, pa.SparseCSFTensor])
def test_sparse_tensor_attrs(sparse_tensor_type):
    data = np.array([[8, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 6]])
    dim_names = ('x', 'y')
    sparse_tensor = sparse_tensor_type.from_dense_numpy(data, dim_names)
    assert sparse_tensor.ndim == 2
    assert sparse_tensor.size == 24
    assert sparse_tensor.shape == data.shape
    assert sparse_tensor.is_mutable
    assert sparse_tensor.dim_name(0) == dim_names[0]
    assert sparse_tensor.dim_names == dim_names
    assert sparse_tensor.non_zero_length == 6
    wr = weakref.ref(sparse_tensor)
    assert wr() is not None
    del sparse_tensor
    assert wr() is None