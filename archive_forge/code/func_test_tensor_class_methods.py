import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
def test_tensor_class_methods():
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 3])
    storage = pa.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], pa.list_(pa.float32(), 6))
    arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    expected = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    result = arr.to_numpy_ndarray()
    np.testing.assert_array_equal(result, expected)
    expected = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    result = arr[:1].to_numpy_ndarray()
    np.testing.assert_array_equal(result, expected)
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.float32, order='C')
    tensor_array_from_numpy = pa.FixedShapeTensorArray.from_numpy_ndarray(arr)
    assert isinstance(tensor_array_from_numpy.type, pa.FixedShapeTensorType)
    assert tensor_array_from_numpy.type.value_type == pa.float32()
    assert tensor_array_from_numpy.type.shape == [2, 3]
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.float32, order='F')
    with pytest.raises(ValueError, match='C-style contiguous segment'):
        pa.FixedShapeTensorArray.from_numpy_ndarray(arr)
    tensor_type = pa.fixed_shape_tensor(pa.int8(), [2, 2, 3], permutation=[0, 2, 1])
    storage = pa.array([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]], pa.list_(pa.int8(), 12))
    arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    with pytest.raises(ValueError, match='non-permuted tensors'):
        arr.to_numpy_ndarray()