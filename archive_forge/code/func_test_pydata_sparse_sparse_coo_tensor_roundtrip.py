import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.skipif(not sparse, reason='requires pydata/sparse')
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_pydata_sparse_sparse_coo_tensor_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([1, 2, 3, 4, 5, 6]).astype(dtype)
    coords = np.array([[0, 0, 2, 3, 1, 3], [0, 2, 0, 4, 5, 5]])
    shape = (4, 6)
    dim_names = ('x', 'y')
    sparse_array = sparse.COO(data=data, coords=coords, shape=shape)
    sparse_tensor = pa.SparseCOOTensor.from_pydata_sparse(sparse_array, dim_names=dim_names)
    out_sparse_array = sparse_tensor.to_pydata_sparse()
    assert sparse_tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert sparse_array.dtype == out_sparse_array.dtype
    assert np.array_equal(sparse_array.data, out_sparse_array.data)
    assert np.array_equal(sparse_array.coords, out_sparse_array.coords)
    assert np.array_equal(sparse_array.todense(), sparse_tensor.to_tensor().to_numpy())