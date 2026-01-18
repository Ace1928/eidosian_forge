import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.parametrize('dtype', complex_dtypes)
@pytest.mark.parametrize('allow_complex', [False, True])
def test_supported_float_dtype_complex(dtype, allow_complex):
    if allow_complex:
        float_dtype = _supported_float_type(dtype, allow_complex=allow_complex)
        if dtype == np.complex64:
            assert float_dtype == np.complex64
        else:
            assert float_dtype == np.complex128
    else:
        with testing.raises(ValueError):
            _supported_float_type(dtype, allow_complex=allow_complex)