import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_rotate07(self, order):
    data = numpy.array([[[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]] * 2, dtype=numpy.float64)
    data = data.transpose()
    expected = numpy.array([[[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]] * 2, dtype=numpy.float64)
    expected = expected.transpose([2, 1, 0])
    out = ndimage.rotate(data, 90, axes=(0, 1), order=order)
    assert_array_almost_equal(out, expected)