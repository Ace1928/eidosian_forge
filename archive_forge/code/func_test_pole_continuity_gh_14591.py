import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_pole_continuity_gh_14591(self):
    u = np.arange(1, 10) * np.pi / 10
    v = np.arange(1, 10) * np.pi / 10
    r = np.zeros((9, 9))
    for p in [(True, True), (True, False), (False, False)]:
        RectSphereBivariateSpline(u, v, r, s=0, pole_continuity=p)