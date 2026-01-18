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
def test_rank08(self):
    array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]])
    expected = [[3, 3, 2, 4, 4], [5, 5, 5, 4, 4], [5, 6, 7, 5, 5]]
    output = ndimage.percentile_filter(array, 50.0, size=(2, 3))
    assert_array_almost_equal(expected, output)
    output = ndimage.rank_filter(array, 3, size=(2, 3))
    assert_array_almost_equal(expected, output)
    output = ndimage.median_filter(array, size=(2, 3))
    assert_array_almost_equal(expected, output)
    with assert_raises(RuntimeError):
        ndimage.percentile_filter(array, 50.0, size=(2, 3), mode=['reflect', 'constant'])
    with assert_raises(RuntimeError):
        ndimage.rank_filter(array, 3, size=(2, 3), mode=['reflect'] * 2)
    with assert_raises(RuntimeError):
        ndimage.median_filter(array, size=(2, 3), mode=['reflect'] * 2)