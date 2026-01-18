import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_preserve_subclass(self):

    def double(row):
        return row * 2

    class MyNDArray(np.ndarray):
        pass
    m = np.array([[0, 1], [2, 3]]).view(MyNDArray)
    expected = np.array([[0, 2], [4, 6]]).view(MyNDArray)
    result = apply_along_axis(double, 0, m)
    assert_(isinstance(result, MyNDArray))
    assert_array_equal(result, expected)
    result = apply_along_axis(double, 1, m)
    assert_(isinstance(result, MyNDArray))
    assert_array_equal(result, expected)