import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
class TestAllFreqConvolves:

    @pytest.mark.parametrize('convapproach', [fftconvolve, oaconvolve])
    def test_invalid_shapes(self, convapproach):
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))
        with assert_raises(ValueError, match="For 'valid' mode, one must be at least as large as the other in every dimension"):
            convapproach(a, b, mode='valid')

    @pytest.mark.parametrize('convapproach', [fftconvolve, oaconvolve])
    def test_invalid_shapes_axes(self, convapproach):
        a = np.zeros([5, 6, 2, 1])
        b = np.zeros([5, 6, 3, 1])
        with assert_raises(ValueError, match='incompatible shapes for in1 and in2: \\(5L?, 6L?, 2L?, 1L?\\) and \\(5L?, 6L?, 3L?, 1L?\\)'):
            convapproach(a, b, axes=[0, 1])

    @pytest.mark.parametrize('a,b', [([1], 2), (1, [2]), ([3], [[2]])])
    @pytest.mark.parametrize('convapproach', [fftconvolve, oaconvolve])
    def test_mismatched_dims(self, a, b, convapproach):
        with assert_raises(ValueError, match='in1 and in2 should have the same dimensionality'):
            convapproach(a, b)

    @pytest.mark.parametrize('convapproach', [fftconvolve, oaconvolve])
    def test_invalid_flags(self, convapproach):
        with assert_raises(ValueError, match="acceptable mode flags are 'valid', 'same', or 'full'"):
            convapproach([1], [2], mode='chips')
        with assert_raises(ValueError, match='when provided, axes cannot be empty'):
            convapproach([1], [2], axes=[])
        with assert_raises(ValueError, match='axes must be a scalar or iterable of integers'):
            convapproach([1], [2], axes=[[1, 2], [3, 4]])
        with assert_raises(ValueError, match='axes must be a scalar or iterable of integers'):
            convapproach([1], [2], axes=[1.0, 2.0, 3.0, 4.0])
        with assert_raises(ValueError, match='axes exceeds dimensionality of input'):
            convapproach([1], [2], axes=[1])
        with assert_raises(ValueError, match='axes exceeds dimensionality of input'):
            convapproach([1], [2], axes=[-2])
        with assert_raises(ValueError, match='all axes must be unique'):
            convapproach([1], [2], axes=[0, 0])

    @pytest.mark.parametrize('dtype', [np.longdouble, np.clongdouble])
    def test_longdtype_input(self, dtype):
        x = np.random.random((27, 27)).astype(dtype)
        y = np.random.random((4, 4)).astype(dtype)
        if np.iscomplexobj(dtype()):
            x += 0.1j
            y -= 0.1j
        res = fftconvolve(x, y)
        assert_allclose(res, convolve(x, y, method='direct'))
        assert res.dtype == dtype