import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_zoom1(self, order):
    for z in [2, [2, 2]]:
        arr = numpy.array(list(range(25))).reshape((5, 5)).astype(float)
        arr = ndimage.zoom(arr, z, order=order)
        assert_equal(arr.shape, (10, 10))
        assert_(numpy.all(arr[-1, :] != 0))
        assert_(numpy.all(arr[-1, :] >= 20 - eps))
        assert_(numpy.all(arr[0, :] <= 5 + eps))
        assert_(numpy.all(arr >= 0 - eps))
        assert_(numpy.all(arr <= 24 + eps))