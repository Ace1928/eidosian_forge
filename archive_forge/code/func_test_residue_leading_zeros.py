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
def test_residue_leading_zeros(self):
    r0, p0, k0 = residue([5, 3, -2, 7], [-4, 0, 8, 3])
    r1, p1, k1 = residue([0, 5, 3, -2, 7], [-4, 0, 8, 3])
    r2, p2, k2 = residue([5, 3, -2, 7], [0, -4, 0, 8, 3])
    r3, p3, k3 = residue([0, 0, 5, 3, -2, 7], [0, 0, 0, -4, 0, 8, 3])
    assert_almost_equal(r0, r1)
    assert_almost_equal(r0, r2)
    assert_almost_equal(r0, r3)
    assert_almost_equal(p0, p1)
    assert_almost_equal(p0, p2)
    assert_almost_equal(p0, p3)
    assert_almost_equal(k0, k1)
    assert_almost_equal(k0, k2)
    assert_almost_equal(k0, k3)