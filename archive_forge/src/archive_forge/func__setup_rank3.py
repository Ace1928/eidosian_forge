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
def _setup_rank3(self, dt):
    a = np.linspace(0, 39, 40).reshape((2, 4, 5), order='F').astype(dt)
    b = np.linspace(0, 23, 24).reshape((2, 3, 4), order='F').astype(dt)
    y_r = array([[[0.0, 184.0, 504.0, 912.0, 1360.0, 888.0, 472.0, 160.0], [46.0, 432.0, 1062.0, 1840.0, 2672.0, 1698.0, 864.0, 266.0], [134.0, 736.0, 1662.0, 2768.0, 3920.0, 2418.0, 1168.0, 314.0], [260.0, 952.0, 1932.0, 3056.0, 4208.0, 2580.0, 1240.0, 332.0], [202.0, 664.0, 1290.0, 1984.0, 2688.0, 1590.0, 712.0, 150.0], [114.0, 344.0, 642.0, 960.0, 1280.0, 726.0, 296.0, 38.0]], [[23.0, 400.0, 1035.0, 1832.0, 2696.0, 1737.0, 904.0, 293.0], [134.0, 920.0, 2166.0, 3680.0, 5280.0, 3306.0, 1640.0, 474.0], [325.0, 1544.0, 3369.0, 5512.0, 7720.0, 4683.0, 2192.0, 535.0], [571.0, 1964.0, 3891.0, 6064.0, 8272.0, 4989.0, 2324.0, 565.0], [434.0, 1360.0, 2586.0, 3920.0, 5264.0, 3054.0, 1312.0, 230.0], [241.0, 700.0, 1281.0, 1888.0, 2496.0, 1383.0, 532.0, 39.0]], [[22.0, 214.0, 528.0, 916.0, 1332.0, 846.0, 430.0, 132.0], [86.0, 484.0, 1098.0, 1832.0, 2600.0, 1602.0, 772.0, 206.0], [188.0, 802.0, 1698.0, 2732.0, 3788.0, 2256.0, 1018.0, 218.0], [308.0, 1006.0, 1950.0, 2996.0, 4052.0, 2400.0, 1078.0, 230.0], [230.0, 692.0, 1290.0, 1928.0, 2568.0, 1458.0, 596.0, 78.0], [126.0, 354.0, 636.0, 924.0, 1212.0, 654.0, 234.0, 0.0]]], dtype=dt)
    return (a, b, y_r)