import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_tensor_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = (100 * np.random.randn(10, 4)).astype(dtype)
    tensor = pa.Tensor.from_numpy(data)
    assert tensor.type == arrow_type
    repr(tensor)
    result = tensor.to_numpy()
    assert (data == result).all()