import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_shift_grid_constant_order1(self):
    x = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    expected_result = numpy.array([[0.25, 0.75, 1.25], [1.25, 3.0, 4.0]])
    assert_array_almost_equal(ndimage.shift(x, (0.5, 0.5), mode='grid-constant', order=1), expected_result)