import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_filter_interp_edges_3d():
    t = np.linspace(-5, 5, 21)
    delta = t[1] - t[0]
    x1 = np.array([t, -t])
    x2 = np.array([t ** 2, 3 * t ** 2 + 5])
    x3 = np.array([t ** 3, 2 * t ** 3 + t ** 2 - 0.5 * t])
    dx1 = np.array([np.ones_like(t), -np.ones_like(t)])
    dx2 = np.array([2 * t, 6 * t])
    dx3 = np.array([3 * t ** 2, 6 * t ** 2 + 2 * t - 0.5])
    z = np.array([x1, x2, x3])
    dz = np.array([dx1, dx2, dx3])
    y = savgol_filter(z, 7, 3, axis=-1, mode='interp', delta=delta)
    assert_allclose(y, z, atol=1e-10)
    dy = savgol_filter(z, 7, 3, axis=-1, mode='interp', deriv=1, delta=delta)
    assert_allclose(dy, dz, atol=1e-10)
    z = np.array([x1.T, x2.T, x3.T])
    dz = np.array([dx1.T, dx2.T, dx3.T])
    y = savgol_filter(z, 7, 3, axis=1, mode='interp', delta=delta)
    assert_allclose(y, z, atol=1e-10)
    dy = savgol_filter(z, 7, 3, axis=1, mode='interp', deriv=1, delta=delta)
    assert_allclose(dy, dz, atol=1e-10)
    z = z.swapaxes(0, 1).copy()
    dz = dz.swapaxes(0, 1).copy()
    y = savgol_filter(z, 7, 3, axis=0, mode='interp', delta=delta)
    assert_allclose(y, z, atol=1e-10)
    dy = savgol_filter(z, 7, 3, axis=0, mode='interp', deriv=1, delta=delta)
    assert_allclose(dy, dz, atol=1e-10)