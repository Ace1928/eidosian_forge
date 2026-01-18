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
def test_rank_1_FIR_init_cond(self):
    x = self.generate((6,))
    b = self.convert_dtype([1, 1, 1])
    a = self.convert_dtype([1])
    zi = self.convert_dtype([1, 1])
    y_r = self.convert_dtype([1, 2, 3, 6, 9, 12.0])
    zf_r = self.convert_dtype([9, 5])
    y, zf = lfilter(b, a, x, zi=zi)
    assert_array_almost_equal(y, y_r)
    assert_array_almost_equal(zf, zf_r)