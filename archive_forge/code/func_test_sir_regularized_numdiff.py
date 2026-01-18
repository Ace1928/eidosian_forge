import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.dimred import (
from numpy.testing import (assert_equal, assert_allclose)
from statsmodels.tools.numdiff import approx_fprime
def test_sir_regularized_numdiff():
    np.random.seed(93482)
    n = 1000
    p = 10
    xmat = np.random.normal(size=(n, p))
    y1 = np.dot(xmat, np.linspace(-1, 1, p))
    y2 = xmat.sum(1)
    y = y2 / (1 + y1 ** 2) + np.random.normal(size=n)
    model = SlicedInverseReg(y, xmat)
    _ = model.fit()
    fmat = np.zeros((p - 2, p))
    for i in range(p - 2):
        fmat[i, i:i + 3] = [1, -2, 1]
    with pytest.warns(UserWarning, match='SIR.fit_regularized did not'):
        _ = model.fit_regularized(2, 3 * fmat)
    for _ in range(5):
        pa = np.random.normal(size=(p, 2))
        pa, _, _ = np.linalg.svd(pa, 0)
        gn = approx_fprime(pa.ravel(), model._regularized_objective, 1e-07)
        gr = model._regularized_grad(pa.ravel())
        assert_allclose(gn, gr, atol=1e-05, rtol=0.0001)