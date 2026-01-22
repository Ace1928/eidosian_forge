import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
class TestNDInterpolators:

    @parametrize_interpolators
    def test_broadcastable_input(self, interpolator):
        np.random.seed(0)
        x = np.random.random(10)
        y = np.random.random(10)
        z = np.hypot(x, y)
        X = np.linspace(min(x), max(x))
        Y = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(X, Y)
        XY = np.vstack((X.ravel(), Y.ravel())).T
        interp = interpolator(list(zip(x, y)), z)
        interp_points0 = interp(XY)
        interp_points1 = interp((X, Y))
        interp_points2 = interp((X, 0.0))
        interp_points3 = interp(X, Y)
        interp_points4 = interp(X, 0.0)
        assert_equal(interp_points0.size == interp_points1.size == interp_points2.size == interp_points3.size == interp_points4.size, True)

    @parametrize_interpolators
    def test_read_only(self, interpolator):
        np.random.seed(0)
        xy = np.random.random((10, 2))
        x, y = (xy[:, 0], xy[:, 1])
        z = np.hypot(x, y)
        XY = np.random.random((50, 2))
        xy.setflags(write=False)
        z.setflags(write=False)
        XY.setflags(write=False)
        interp = interpolator(xy, z)
        interp(XY)