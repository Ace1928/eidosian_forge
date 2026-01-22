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
class TestSplev:

    def test_1d_shape(self):
        x = [1, 2, 3, 4, 5]
        y = [4, 5, 6, 7, 8]
        tck = splrep(x, y)
        z = splev([1], tck)
        assert_equal(z.shape, (1,))
        z = splev(1, tck)
        assert_equal(z.shape, ())

    def test_2d_shape(self):
        x = [1, 2, 3, 4, 5]
        y = [4, 5, 6, 7, 8]
        tck = splrep(x, y)
        t = np.array([[1.0, 1.5, 2.0, 2.5], [3.0, 3.5, 4.0, 4.5]])
        z = splev(t, tck)
        z0 = splev(t[0], tck)
        z1 = splev(t[1], tck)
        assert_equal(z, np.vstack((z0, z1)))

    def test_extrapolation_modes(self):
        x = [1, 2, 3]
        y = [0, 2, 4]
        tck = splrep(x, y, k=1)
        rstl = [[-2, 6], [0, 0], None, [0, 4]]
        for ext in (0, 1, 3):
            assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])
        assert_raises(ValueError, splev, [0, 4], tck, ext=2)