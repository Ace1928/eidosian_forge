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
def test_valid_origins():
    """Regression test for #1311."""

    def func(x):
        return numpy.mean(x)
    data = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float64)
    assert_raises(ValueError, ndimage.generic_filter, data, func, size=3, origin=2)
    assert_raises(ValueError, ndimage.generic_filter1d, data, func, filter_size=3, origin=2)
    assert_raises(ValueError, ndimage.percentile_filter, data, 0.2, size=3, origin=2)
    for filter in [ndimage.uniform_filter, ndimage.minimum_filter, ndimage.maximum_filter, ndimage.maximum_filter1d, ndimage.median_filter, ndimage.minimum_filter1d]:
        list(filter(data, 3, origin=-1))
        list(filter(data, 3, origin=1))
        assert_raises(ValueError, filter, data, 3, origin=2)