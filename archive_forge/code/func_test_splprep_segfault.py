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
def test_splprep_segfault():
    t = np.arange(0, 1.1, 0.1)
    x = np.sin(2 * np.pi * t)
    y = np.cos(2 * np.pi * t)
    tck, u = splprep([x, y], s=0)
    np.arange(0, 1.01, 0.01)
    uknots = tck[0]
    tck, u = splprep([x, y], task=-1, t=uknots)