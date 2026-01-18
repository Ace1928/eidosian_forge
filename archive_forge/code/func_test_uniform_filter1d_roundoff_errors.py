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
def test_uniform_filter1d_roundoff_errors():
    in_ = numpy.repeat([0, 1, 0], [9, 9, 9])
    for filter_size in range(3, 10):
        out = ndimage.uniform_filter1d(in_, filter_size)
        assert_equal(out.sum(), 10 - filter_size)