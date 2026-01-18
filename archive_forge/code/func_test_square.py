import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_square(self):
    points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
    values = np.array([1.0, 2.0, -3.0, 5.0], dtype=np.float64)

    def ip(x, y):
        t1 = x + y <= 1
        t2 = ~t1
        x1 = x[t1]
        y1 = y[t1]
        x2 = x[t2]
        y2 = y[t2]
        z = 0 * x
        z[t1] = values[0] * (1 - x1 - y1) + values[1] * y1 + values[3] * x1
        z[t2] = values[2] * (x2 + y2 - 1) + values[1] * (1 - x2) + values[3] * (1 - y2)
        return z
    xx, yy = np.broadcast_arrays(np.linspace(0, 1, 14)[:, None], np.linspace(0, 1, 14)[None, :])
    xx = xx.ravel()
    yy = yy.ravel()
    xi = np.array([xx, yy]).T.copy()
    zi = interpnd.LinearNDInterpolator(points, values)(xi)
    assert_almost_equal(zi, ip(xx, yy))