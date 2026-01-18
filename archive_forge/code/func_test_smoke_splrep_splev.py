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
def test_smoke_splrep_splev(self):
    self.check_1(s=1e-06)
    self.check_1(b=1.5 * np.pi)
    self.check_1(b=1.5 * np.pi, xe=2 * np.pi, per=1, s=0.1)