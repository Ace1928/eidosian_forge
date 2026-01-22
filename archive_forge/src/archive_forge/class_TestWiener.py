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
class TestWiener:

    def test_basic(self):
        g = array([[5, 6, 4, 3], [3, 5, 6, 2], [2, 3, 5, 6], [1, 6, 9, 7]], 'd')
        h = array([[2.16374269, 3.2222222222, 2.8888888889, 1.6666666667], [2.666666667, 4.33333333333, 4.44444444444, 2.8888888888], [2.222222222, 4.4444444444, 5.4444444444, 4.801066874837], [1.33333333333, 3.92735042735, 6.0712560386, 5.0404040404]])
        assert_array_almost_equal(signal.wiener(g), h, decimal=6)
        assert_array_almost_equal(signal.wiener(g, mysize=3), h, decimal=6)