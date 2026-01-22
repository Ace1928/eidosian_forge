import itertools
import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate._fitpack2 import dfitpack_int
class TestBisplrep:

    def test_overflow(self):
        from numpy.lib.stride_tricks import as_strided
        if dfitpack_int.itemsize == 8:
            size = 1500000 ** 2
        else:
            size = 400 ** 2
        x = as_strided(np.zeros(()), shape=(size,))
        assert_raises(OverflowError, bisplrep, x, x, x, w=x, xb=0, xe=1, yb=0, ye=1, s=0)

    def test_regression_1310(self):
        data = np.load(data_file('bug-1310.npz'))['data']
        bisplrep(data[:, 0], data[:, 1], data[:, 2], kx=3, ky=3, s=0, full_output=True)

    @pytest.mark.skipif(dfitpack_int != np.int64, reason='needs ilp64 fitpack')
    def test_ilp64_bisplrep(self):
        check_free_memory(28000)
        x = np.linspace(0, 1, 400)
        y = np.linspace(0, 1, 400)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        tck = bisplrep(x, y, z, kx=3, ky=3, s=0)
        assert_allclose(bisplev(0.5, 0.5, tck), 0.0)