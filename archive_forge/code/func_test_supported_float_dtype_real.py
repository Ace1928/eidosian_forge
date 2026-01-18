import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.parametrize('dtype', [bool, np.float16, np.float32, np.float64, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64])
def test_supported_float_dtype_real(dtype):
    float_dtype = _supported_float_type(dtype)
    if dtype in [np.float16, np.float32]:
        assert float_dtype == np.float32
    else:
        assert float_dtype == np.float64