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
def test_splder_vs_splev(self):
    for n in range(3 + 1):
        xx = np.linspace(-1, 2, 2000)
        if n == 3:
            xx = xx[(xx >= 0) & (xx <= 1)]
        dy = splev(xx, self.spl, n)
        spl2 = splder(self.spl, n)
        dy2 = splev(xx, spl2)
        if n == 1:
            assert_allclose(dy, dy2, rtol=2e-06)
        else:
            assert_allclose(dy, dy2)