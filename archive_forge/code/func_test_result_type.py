from __future__ import annotations
import numpy as np
import pytest
from xarray.core import dtypes
@pytest.mark.parametrize('args, expected', [([bool], bool), ([bool, np.bytes_], np.object_), ([np.float32, np.float64], np.float64), ([np.float32, np.bytes_], np.object_), ([np.str_, np.int64], np.object_), ([np.str_, np.str_], np.str_), ([np.bytes_, np.str_], np.object_)])
def test_result_type(args, expected) -> None:
    actual = dtypes.result_type(*args)
    assert actual == expected