import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_rotate06(self, order):
    data = numpy.empty((3, 4, 3))
    for i in range(3):
        data[:, :, i] = numpy.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=numpy.float64)
    expected = numpy.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=numpy.float64)
    out = ndimage.rotate(data, 90, order=order)
    for i in range(3):
        assert_array_almost_equal(out[:, :, i], expected)