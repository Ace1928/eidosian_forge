import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
def test_orders_gauss():
    arr = numpy.zeros((1,))
    assert_equal(0, ndimage.gaussian_filter(arr, 1, order=0))
    assert_equal(0, ndimage.gaussian_filter(arr, 1, order=3))
    assert_raises(ValueError, ndimage.gaussian_filter, arr, 1, -1)
    assert_equal(0, ndimage.gaussian_filter1d(arr, 1, axis=-1, order=0))
    assert_equal(0, ndimage.gaussian_filter1d(arr, 1, axis=-1, order=3))
    assert_raises(ValueError, ndimage.gaussian_filter1d, arr, 1, -1, -1)