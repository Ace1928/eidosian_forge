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
def test_sine(self):
    rate = 2000
    t = np.linspace(0, 1.0, rate + 1)
    xlow = np.sin(5 * 2 * np.pi * t)
    xhigh = np.sin(250 * 2 * np.pi * t)
    x = xlow + xhigh
    zpk = butter(8, 0.125, output='zpk')
    r = np.abs(zpk[1]).max()
    eps = 1e-05
    n = int(np.ceil(np.log(eps) / np.log(r)))
    y = self.filtfilt(zpk, x, padlen=n)
    err = np.abs(y - xlow).max()
    assert_(err < 0.0001)
    x2d = np.vstack([xlow, xlow + xhigh])
    y2d = self.filtfilt(zpk, x2d, padlen=n, axis=1)
    assert_equal(y2d.shape, x2d.shape)
    err = np.abs(y2d - xlow).max()
    assert_(err < 0.0001)
    y2dt = self.filtfilt(zpk, x2d.T, padlen=n, axis=0)
    assert_equal(y2d, y2dt.T)