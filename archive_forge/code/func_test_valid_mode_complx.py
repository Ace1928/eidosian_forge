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
def test_valid_mode_complx(self):
    e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
    f = np.array([[1, 2, 3], [3, 4, 5]], dtype=complex) + 1j
    h = array([[62.0 + 24j, 80.0 + 30j, 98.0 + 36j, 116.0 + 42j, 134.0 + 48j]])
    g = convolve2d(e, f, 'valid')
    assert_array_almost_equal(g, h)
    g = convolve2d(f, e, 'valid')
    assert_array_equal(g, h)