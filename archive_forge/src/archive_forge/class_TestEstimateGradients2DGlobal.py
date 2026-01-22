import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
class TestEstimateGradients2DGlobal:

    def test_smoketest(self):
        x = np.array([(0, 0), (0, 2), (1, 0), (1, 2), (0.25, 0.75), (0.6, 0.8)], dtype=float)
        tri = qhull.Delaunay(x)
        funcs = [(lambda x, y: 0 * x + 1, (0, 0)), (lambda x, y: 0 + x, (1, 0)), (lambda x, y: -2 + y, (0, 1)), (lambda x, y: 3 + 3 * x + 14.15 * y, (3, 14.15))]
        for j, (func, grad) in enumerate(funcs):
            z = func(x[:, 0], x[:, 1])
            dz = interpnd.estimate_gradients_2d_global(tri, z, tol=1e-06)
            assert_equal(dz.shape, (6, 2))
            assert_allclose(dz, np.array(grad)[None, :] + 0 * dz, rtol=1e-05, atol=1e-05, err_msg='item %d' % j)

    def test_regression_2359(self):
        points = np.load(data_file('estimate_gradients_hang.npy'))
        values = np.random.rand(points.shape[0])
        tri = qhull.Delaunay(points)
        with suppress_warnings() as sup:
            sup.filter(interpnd.GradientEstimationWarning, 'Gradient estimation did not converge')
            interpnd.estimate_gradients_2d_global(tri, values, maxiter=1)