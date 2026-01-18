import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_qspline1d_eval(self):
    np.random.seed(12465)
    assert_allclose(bsp.qspline1d_eval(array([0.0, 0]), [0.0]), array([0.0]))
    assert_array_equal(bsp.qspline1d_eval(array([1.0, 0, 1]), []), array([]))
    x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    dx = x[1] - x[0]
    newx = [-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
    y = array([4.216, 6.864, 3.514, 6.203, 6.759, 7.433, 7.874, 5.879, 1.396, 4.094])
    cj = bsp.qspline1d(y)
    newy = array([6.203, 4.49418159, 3.514, 5.18390821, 6.864, 5.91436915, 4.21600002, 5.91436915, 6.864, 5.18390821, 3.514, 4.49418159, 6.203, 6.71900226, 6.759, 7.03980488, 7.433, 7.81016848, 7.874, 7.32718426, 5.879, 3.23872593, 1.396, 2.34046013, 4.094, 2.34046013, 1.396, 3.23872593, 5.879, 7.32718426, 7.874, 7.81016848, 7.433, 7.03980488, 6.759, 6.71900226, 6.203, 4.49418159])
    assert_allclose(bsp.qspline1d_eval(cj, newx, dx=dx, x0=x[0]), newy)