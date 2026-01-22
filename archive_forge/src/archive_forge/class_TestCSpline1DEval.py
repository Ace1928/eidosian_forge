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
class TestCSpline1DEval:

    def test_basic(self):
        y = array([1, 2, 3, 4, 3, 2, 1, 2, 3.0])
        x = arange(len(y))
        dx = x[1] - x[0]
        cj = signal.cspline1d(y)
        x2 = arange(len(y) * 10.0) / 10.0
        y2 = signal.cspline1d_eval(cj, x2, dx=dx, x0=x[0])
        assert_array_almost_equal(y2[::10], y, decimal=5)

    def test_complex(self):
        x = np.arange(2)
        y = np.zeros(x.shape, dtype=np.complex64)
        T = 10.0
        f = 1.0 / T
        y = np.exp(2j * np.pi * f * x)
        cy = signal.cspline1d(y)
        xnew = np.array([0.5])
        ynew = signal.cspline1d_eval(cy, xnew)
        assert_equal(ynew.dtype, y.dtype)