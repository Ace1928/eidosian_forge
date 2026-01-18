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
def test_rank0(self, dt):
    a = np.array(np.random.randn()).astype(dt)
    a += 1j * np.array(np.random.randn()).astype(dt)
    b = np.array(np.random.randn()).astype(dt)
    b += 1j * np.array(np.random.randn()).astype(dt)
    y_r = (correlate(a.real, b.real) + correlate(a.imag, b.imag)).astype(dt)
    y_r += 1j * np.array(-correlate(a.real, b.imag) + correlate(a.imag, b.real))
    y = correlate(a, b, 'full')
    assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
    assert_equal(y.dtype, dt)
    assert_equal(correlate([1], [2j]), correlate(1, 2j))
    assert_equal(correlate([2j], [3j]), correlate(2j, 3j))
    assert_equal(correlate([3j], [4]), correlate(3j, 4))