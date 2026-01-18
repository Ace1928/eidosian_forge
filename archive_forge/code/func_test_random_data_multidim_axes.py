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
@pytest.mark.parametrize('axes', [[1, 4], [4, 1], [1, -1], [-1, 1], [-4, 4], [4, -4], [-4, -1], [-1, -4]])
def test_random_data_multidim_axes(self, axes):
    a_shape, b_shape = ((123, 22), (132, 11))
    np.random.seed(1234)
    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    expected = convolve2d(a, b, 'full')
    a = a[:, :, None, None, None]
    b = b[:, :, None, None, None]
    expected = expected[:, :, None, None, None]
    a = np.moveaxis(a.swapaxes(0, 2), 1, 4)
    b = np.moveaxis(b.swapaxes(0, 2), 1, 4)
    expected = np.moveaxis(expected.swapaxes(0, 2), 1, 4)
    a = np.tile(a, [2, 1, 3, 1, 1])
    b = np.tile(b, [2, 1, 1, 4, 1])
    expected = np.tile(expected, [2, 1, 3, 4, 1])
    out = fftconvolve(a, b, 'full', axes=axes)
    assert_allclose(out, expected, rtol=1e-10, atol=1e-10)