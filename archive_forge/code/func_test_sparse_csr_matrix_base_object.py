import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
def test_sparse_csr_matrix_base_object():
    data = np.array([[8, 2, 5, 3, 4, 6]]).T
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    array = np.array([[8, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 6]])
    sparse_tensor = pa.SparseCSRMatrix.from_dense_numpy(array)
    n = sys.getrefcount(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sys.getrefcount(sparse_tensor) == n + 3
    sparse_tensor = None
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)