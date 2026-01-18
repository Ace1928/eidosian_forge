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
def test_splev_der_k():
    tck = (np.array([0.0, 0.0, 2.5, 2.5]), np.array([-1.56679978, 2.43995873, 0.0, 0.0]), 1)
    t, c, k = tck
    x = np.array([-3, 0, 2.5, 3])
    assert_allclose(splev(x, tck), c[0] + (c[1] - c[0]) * x / t[2])
    assert_allclose(splev(x, tck, 1), (c[1] - c[0]) / t[2])
    np.random.seed(1234)
    x = np.sort(np.random.random(30))
    y = np.random.random(30)
    t, c, k = splrep(x, y)
    x = [t[0] - 1.0, t[-1] + 1.0]
    tck2 = splder((t, c, k), k)
    assert_allclose(splev(x, (t, c, k), k), splev(x, tck2))