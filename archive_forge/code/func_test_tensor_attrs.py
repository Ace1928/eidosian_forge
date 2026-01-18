import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_attrs():
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)
    assert tensor.ndim == 2
    assert tensor.dim_names == []
    assert tensor.size == 40
    assert tensor.shape == data.shape
    assert tensor.strides == data.strides
    assert tensor.is_contiguous
    assert tensor.is_mutable
    data2 = data.copy()
    data2.flags.writeable = False
    tensor = pa.Tensor.from_numpy(data2)
    assert not tensor.is_mutable
    tensor = pa.Tensor.from_numpy(data, dim_names=('x', 'y'))
    assert tensor.ndim == 2
    assert tensor.dim_names == ['x', 'y']
    assert tensor.dim_name(0) == 'x'
    assert tensor.dim_name(1) == 'y'
    wr = weakref.ref(tensor)
    assert wr() is not None
    del tensor
    assert wr() is None