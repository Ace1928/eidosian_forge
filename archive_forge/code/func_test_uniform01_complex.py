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
def test_uniform01_complex(self):
    array = numpy.array([2 + 1j, 4 + 2j, 6 + 3j], dtype=numpy.complex128)
    size = 2
    output = ndimage.uniform_filter1d(array, size, origin=-1)
    assert_array_almost_equal([3, 5, 6], output.real)
    assert_array_almost_equal([1.5, 2.5, 3], output.imag)