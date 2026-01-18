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
@pytest.mark.parametrize('func', (sosfilt, lfilter))
def test_nonnumeric_dtypes(func):
    x = [Decimal(1), Decimal(2), Decimal(3)]
    b = [Decimal(1), Decimal(2), Decimal(3)]
    a = [Decimal(1), Decimal(2), Decimal(3)]
    x = np.array(x)
    assert x.dtype.kind == 'O'
    desired = lfilter(np.array(b, float), np.array(a, float), x.astype(float))
    if func is sosfilt:
        actual = sosfilt([b + a], x)
    else:
        actual = lfilter(b, a, x)
    assert all((isinstance(x, Decimal) for x in actual))
    assert_allclose(actual.astype(float), desired.astype(float))
    if func is lfilter:
        args = [1.0, 1.0]
    else:
        args = [tf2sos(1.0, 1.0)]
    with pytest.raises(ValueError, match='must be at least 1-D'):
        func(*args, x=1.0)