import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_shift08(self, order):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])
    out = ndimage.shift(data, [1, 1], order=order)
    assert_array_almost_equal(out, [[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]])