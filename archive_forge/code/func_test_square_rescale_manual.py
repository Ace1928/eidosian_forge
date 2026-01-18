import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_square_rescale_manual(self):
    points = np.array([(0, 0), (0, 100), (10, 100), (10, 0), (1, 5)], dtype=np.float64)
    points_rescaled = np.array([(0, 0), (0, 1), (1, 1), (1, 0), (0.1, 0.05)], dtype=np.float64)
    values = np.array([1.0, 2.0, -3.0, 5.0, 9.0], dtype=np.float64)
    xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:, None], np.linspace(0, 100, 14)[None, :])
    xx = xx.ravel()
    yy = yy.ravel()
    xi = np.array([xx, yy]).T.copy()
    for method in ('nearest', 'linear', 'cubic'):
        msg = method
        zi = griddata(points_rescaled, values, xi / np.array([10, 100.0]), method=method)
        zi_rescaled = griddata(points, values, xi, method=method, rescale=True)
        assert_allclose(zi, zi_rescaled, err_msg=msg, atol=1e-12)