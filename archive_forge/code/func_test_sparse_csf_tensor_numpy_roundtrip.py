import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csf_tensor_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = [np.array([0, 2, 3, 4, 6])]
    indices = [np.array([0, 1, 2, 3]), np.array([0, 2, 5, 0, 4, 5])]
    axis_order = (0, 1)
    shape = (4, 6)
    dim_names = ('x', 'y')
    sparse_tensor = pa.SparseCSFTensor.from_numpy(data, indptr, indices, shape, axis_order, dim_names)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr[0], result_indptr[0])
    assert np.array_equal(indices[0], result_indices[0])
    assert np.array_equal(indices[1], result_indices[1])
    assert sparse_tensor.dim_names == dim_names