import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_regression_2359(self):
    points = np.load(data_file('estimate_gradients_hang.npy'))
    values = np.random.rand(points.shape[0])
    tri = qhull.Delaunay(points)
    with suppress_warnings() as sup:
        sup.filter(interpnd.GradientEstimationWarning, 'Gradient estimation did not converge')
        interpnd.estimate_gradients_2d_global(tri, values, maxiter=1)