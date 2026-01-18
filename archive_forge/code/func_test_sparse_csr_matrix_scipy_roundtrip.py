import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.skipif(not csr_matrix, reason='requires scipy')
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csr_matrix_scipy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([8, 2, 5, 3, 4, 6]).astype(dtype)
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    shape = (4, 6)
    dim_names = ('x', 'y')
    sparse_array = csr_matrix((data, indices, indptr), shape=shape)
    sparse_tensor = pa.SparseCSRMatrix.from_scipy(sparse_array, dim_names=dim_names)
    out_sparse_array = sparse_tensor.to_scipy()
    assert sparse_tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert sparse_array.dtype == out_sparse_array.dtype
    assert np.array_equal(sparse_array.data, out_sparse_array.data)
    assert np.array_equal(sparse_array.indptr, out_sparse_array.indptr)
    assert np.array_equal(sparse_array.indices, out_sparse_array.indices)
    if dtype_str == 'f2':
        dense_array = sparse_array.astype(np.float32).toarray().astype(np.float16)
    else:
        dense_array = sparse_array.toarray()
    assert np.array_equal(dense_array, sparse_tensor.to_tensor().to_numpy())