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
def test_rank06(self):
    array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]])
    expected = [[2, 2, 1, 1, 1], [3, 3, 2, 1, 1], [5, 5, 3, 3, 1]]
    output = ndimage.rank_filter(array, 1, size=[2, 3])
    assert_array_almost_equal(expected, output)
    output = ndimage.percentile_filter(array, 17, size=(2, 3))
    assert_array_almost_equal(expected, output)