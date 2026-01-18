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
def test_rank03(self):
    array = numpy.array([3, 2, 5, 1, 4])
    output = ndimage.rank_filter(array, 1, size=[2])
    assert_array_almost_equal([3, 3, 5, 5, 4], output)
    output = ndimage.percentile_filter(array, 100, size=2)
    assert_array_almost_equal([3, 3, 5, 5, 4], output)