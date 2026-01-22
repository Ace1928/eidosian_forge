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
def _validate_complex(self, array, kernel, type2, mode='reflect', cval=0):
    real_dtype = numpy.asarray([], dtype=type2).real.dtype
    expected = _complex_correlate(array, kernel, real_dtype, convolve=False, mode=mode, cval=cval)
    if array.ndim == 1:
        correlate = functools.partial(ndimage.correlate1d, axis=-1, mode=mode, cval=cval)
        convolve = functools.partial(ndimage.convolve1d, axis=-1, mode=mode, cval=cval)
    else:
        correlate = functools.partial(ndimage.correlate, mode=mode, cval=cval)
        convolve = functools.partial(ndimage.convolve, mode=mode, cval=cval)
    output = correlate(array, kernel, output=type2)
    assert_array_almost_equal(expected, output)
    assert_equal(output.dtype.type, type2)
    output = numpy.zeros_like(array, dtype=type2)
    correlate(array, kernel, output=output)
    assert_array_almost_equal(expected, output)
    output = convolve(array, kernel, output=type2)
    expected = _complex_correlate(array, kernel, real_dtype, convolve=True, mode=mode, cval=cval)
    assert_array_almost_equal(expected, output)
    assert_equal(output.dtype.type, type2)
    convolve(array, kernel, output=output)
    assert_array_almost_equal(expected, output)
    assert_equal(output.dtype.type, type2)
    with pytest.warns(UserWarning, match='promoting specified output dtype to complex'):
        correlate(array, kernel, output=real_dtype)
    with pytest.warns(UserWarning, match='promoting specified output dtype to complex'):
        convolve(array, kernel, output=real_dtype)
    output_real = numpy.zeros_like(array, dtype=real_dtype)
    with assert_raises(RuntimeError):
        correlate(array, kernel, output=output_real)
    with assert_raises(RuntimeError):
        convolve(array, kernel, output=output_real)