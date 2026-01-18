import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_rotate_exact_180(self):
    a = numpy.tile(numpy.arange(5), (5, 1))
    b = ndimage.rotate(ndimage.rotate(a, 180), -180)
    assert_equal(a, b)