import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.dimred import (
from numpy.testing import (assert_equal, assert_allclose)
from statsmodels.tools.numdiff import approx_fprime
def test_sir_regularized_2d():
    np.random.seed(93482)
    n = 1000
    p = 10
    xmat = np.random.normal(size=(n, p))
    y1 = np.dot(xmat[:, 0:4], np.r_[1, 1, -1, -1])
    y2 = np.dot(xmat[:, 4:8], np.r_[1, 1, -1, -1])
    y = y1 + np.arctan(y2) + np.random.normal(size=n)
    model = SlicedInverseReg(y, xmat)
    rslt1 = model.fit()
    fmat = np.zeros((1, p))
    for d in (1, 2, 3, 4):
        if d < 3:
            rslt2 = model.fit_regularized(d, fmat)
        else:
            with pytest.warns(UserWarning, match='SIR.fit_regularized did'):
                rslt2 = model.fit_regularized(d, fmat)
        pa1 = rslt1.params[:, 0:d]
        pa1, _, _ = np.linalg.svd(pa1, 0)
        pa2 = rslt2.params
        _, s, _ = np.linalg.svd(np.dot(pa1.T, pa2))
        assert_allclose(np.sum(s), d, atol=0.1, rtol=0.1)