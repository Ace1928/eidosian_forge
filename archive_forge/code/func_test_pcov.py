import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def test_pcov(self):
    xdata = np.array([0, 1, 2, 3, 4, 5])
    ydata = np.array([1, 1, 5, 7, 8, 12])
    sigma = np.array([1, 2, 1, 2, 1, 2])

    def f(x, a, b):
        return a * x + b
    for method in ['lm', 'trf', 'dogbox']:
        popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma, method=method)
        perr_scaled = np.sqrt(np.diag(pcov))
        assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=0.001)
        popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3 * sigma, method=method)
        perr_scaled = np.sqrt(np.diag(pcov))
        assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=0.001)
        popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma, absolute_sigma=True, method=method)
        perr = np.sqrt(np.diag(pcov))
        assert_allclose(perr, [0.30714756, 0.85045308], rtol=0.001)
        popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3 * sigma, absolute_sigma=True, method=method)
        perr = np.sqrt(np.diag(pcov))
        assert_allclose(perr, [3 * 0.30714756, 3 * 0.85045308], rtol=0.001)

    def f_flat(x, a, b):
        return a * x
    pcov_expected = np.array([np.inf] * 4).reshape(2, 2)
    with suppress_warnings() as sup:
        sup.filter(OptimizeWarning, 'Covariance of the parameters could not be estimated')
        popt, pcov = curve_fit(f_flat, xdata, ydata, p0=[2, 0], sigma=sigma)
        popt1, pcov1 = curve_fit(f, xdata[:2], ydata[:2], p0=[2, 0])
    assert_(pcov.shape == (2, 2))
    assert_array_equal(pcov, pcov_expected)
    assert_(pcov1.shape == (2, 2))
    assert_array_equal(pcov1, pcov_expected)