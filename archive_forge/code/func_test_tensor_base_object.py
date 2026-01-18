import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_base_object():
    tensor = pa.Tensor.from_numpy(np.random.randn(10, 4))
    n = sys.getrefcount(tensor)
    array = tensor.to_numpy()
    assert sys.getrefcount(tensor) == n + 1