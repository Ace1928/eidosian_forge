import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_zoom_infinity(self):
    dim = 8
    ndimage.zoom(numpy.zeros((dim, dim)), 1.0 / dim, mode='nearest')