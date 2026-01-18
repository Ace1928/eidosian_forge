import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_coo_tensor_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[1, 2, 3, 4, 5, 6]]).T.astype(dtype)
    coords = np.array([[0, 0, 2, 3, 1, 3], [0, 2, 0, 4, 5, 5]]).T
    shape = (4, 6)
    dim_names = ('x', 'y')
    sparse_tensor = pa.SparseCOOTensor.from_numpy(data, coords, shape, dim_names)
    repr(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(coords, result_coords)
    assert sparse_tensor.dim_names == dim_names