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
def test_rank1(self, dt):
    x = np.linspace(0, 5, 6).astype(dt)
    b = np.array([1, -1]).astype(dt)
    a = np.array([0.5, -0.5]).astype(dt)
    y_r = np.array([0, 2, 4, 6, 8, 10.0]).astype(dt)
    sos = tf2sos(b, a)
    assert_array_almost_equal(sosfilt(tf2sos(b, a), x), y_r)
    b = np.array([1, 1]).astype(dt)
    a = np.array([1, 0]).astype(dt)
    y_r = np.array([0, 1, 3, 5, 7, 9.0]).astype(dt)
    assert_array_almost_equal(sosfilt(tf2sos(b, a), x), y_r)
    b = [1, 1, 0]
    a = [1, 0, 0]
    x = np.ones(8)
    sos = np.concatenate((b, a))
    sos.shape = (1, 6)
    y = sosfilt(sos, x)
    assert_allclose(y, [1, 2, 2, 2, 2, 2, 2, 2])