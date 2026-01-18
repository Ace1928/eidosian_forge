import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.dimred import (
from numpy.testing import (assert_equal, assert_allclose)
from statsmodels.tools.numdiff import approx_fprime
def test_sir_regularized_1d():
    np.random.seed(93482)
    n = 1000
    p = 10
    xmat = np.random.normal(size=(n, p))
    y = np.dot(xmat[:, 0:4], np.r_[1, 1, -1, -1]) + np.random.normal(size=n)
    model = SlicedInverseReg(y, xmat)
    rslt = model.fit()
    fmat = np.zeros((2, p))
    fmat[0, 0:2] = [1, -1]
    fmat[1, 2:4] = [1, -1]
    rslt2 = model.fit_regularized(1, 3 * fmat)
    pa0 = np.zeros(p)
    pa0[0:4] = [1, 1, -1, -1]
    pa1 = rslt.params[:, 0]
    pa2 = rslt2.params[:, 0:2]

    def sim(x, y):
        x = x / np.sqrt(np.sum(x * x))
        y = y / np.sqrt(np.sum(y * y))
        return 1 - np.abs(np.dot(x, y))
    assert_equal(sim(pa0, pa1) > sim(pa0, pa2), True)
    assert_equal(sim(pa0, pa2) < 0.001, True)
    assert_equal(np.sum(np.dot(fmat, pa1) ** 2) > np.sum(np.dot(fmat, pa2) ** 2), True)