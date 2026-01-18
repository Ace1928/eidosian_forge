import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_shift_reflect(self, order):
    x = numpy.array([[0, 1, 2], [3, 4, 5]])
    assert_array_almost_equal(ndimage.shift(x, x.shape, mode='reflect', order=order), x[::-1, ::-1])