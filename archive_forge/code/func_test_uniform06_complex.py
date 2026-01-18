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
@pytest.mark.parametrize('dtype_array', complex_types)
@pytest.mark.parametrize('dtype_output', complex_types)
def test_uniform06_complex(self, dtype_array, dtype_output):
    filter_shape = [2, 2]
    array = numpy.array([[4, 8 + 5j, 12], [16, 20, 24]], dtype_array)
    output = ndimage.uniform_filter(array, filter_shape, output=dtype_output)
    assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output.real)
    assert_equal(output.dtype.type, dtype_output)