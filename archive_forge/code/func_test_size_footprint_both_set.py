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
def test_size_footprint_both_set():
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'ignoring size because footprint is set')
        arr = numpy.random.random((10, 20, 30))
        ndimage.rank_filter(arr, 5, size=2, footprint=numpy.ones((1, 1, 10), dtype=bool))