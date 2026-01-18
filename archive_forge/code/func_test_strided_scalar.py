import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_strided_scalar():
    for shape, scalar in product(((2,), (2, 3), (2, 3, 4)), (1, 2, np.int16(3))):
        expected = np.zeros(shape, dtype=np.array(scalar).dtype) + scalar
        observed = strided_scalar(shape, scalar)
        assert_array_equal(observed, expected)
        assert observed.shape == shape
        assert observed.dtype == expected.dtype
        assert_array_equal(observed.strides, 0)
        assert not observed.flags.writeable

        def setval(x):
            x[..., 0] = 99
        with pytest.raises((RuntimeError, ValueError)):
            setval(observed)
    assert_array_equal(strided_scalar((2, 3, 4)), np.zeros((2, 3, 4)))