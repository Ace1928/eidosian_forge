import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_not_increasing_input(self):
    NSamp = 20
    Theta = np.random.uniform(0, np.pi, NSamp)
    Phi = np.random.uniform(0, 2 * np.pi, NSamp)
    Data = np.ones(NSamp)
    Interpolator = SmoothSphereBivariateSpline(Theta, Phi, Data, s=3.5)
    NLon = 6
    NLat = 3
    GridPosLats = np.arange(NLat) / NLat * np.pi
    GridPosLons = np.arange(NLon) / NLon * 2 * np.pi
    Interpolator(GridPosLats, GridPosLons)
    nonGridPosLats = GridPosLats.copy()
    nonGridPosLats[2] = 0.001
    with assert_raises(ValueError) as exc_info:
        Interpolator(nonGridPosLats, GridPosLons)
    assert 'x must be strictly increasing' in str(exc_info.value)
    nonGridPosLons = GridPosLons.copy()
    nonGridPosLons[2] = 0.001
    with assert_raises(ValueError) as exc_info:
        Interpolator(GridPosLats, nonGridPosLons)
    assert 'y must be strictly increasing' in str(exc_info.value)