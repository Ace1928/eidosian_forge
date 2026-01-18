import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_zoom2(self):
    arr = numpy.arange(12).reshape((3, 4))
    out = ndimage.zoom(ndimage.zoom(arr, 2), 0.5)
    assert_array_equal(out, arr)