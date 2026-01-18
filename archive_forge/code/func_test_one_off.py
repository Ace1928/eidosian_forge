import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_one_off():
    x = np.array([[1, 2, 3]])
    y = np.array([[1], [2], [3]])
    bx, by = broadcast_arrays(x, y)
    bx0 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    by0 = bx0.T
    assert_array_equal(bx0, bx)
    assert_array_equal(by0, by)