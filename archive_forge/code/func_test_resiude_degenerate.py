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
def test_resiude_degenerate(self):
    r, p, k = residue([0, 0], [1, 6, 8])
    assert_almost_equal(r, [0, 0])
    assert_almost_equal(p, [-2, -4])
    assert_equal(k.size, 0)
    r, p, k = residue(0, 1)
    assert_equal(r.size, 0)
    assert_equal(p.size, 0)
    assert_equal(k.size, 0)
    with pytest.raises(ValueError, match='Denominator `a` is zero.'):
        residue(1, 0)