import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_zoom3(self):
    arr = numpy.array([[1, 2]])
    out1 = ndimage.zoom(arr, (2, 1))
    out2 = ndimage.zoom(arr, (1, 2))
    assert_array_almost_equal(out1, numpy.array([[1, 2], [1, 2]]))
    assert_array_almost_equal(out2, numpy.array([[1, 1, 2, 2]]))