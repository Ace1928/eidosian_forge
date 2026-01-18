from statsmodels.compat.platform import PLATFORM_OSX
from statsmodels.regression.process_regression import (
import numpy as np
import pandas as pd
import pytest
import collections
import statsmodels.tools.numdiff as nd
from numpy.testing import assert_allclose, assert_equal
@pytest.mark.parametrize('noise', [False, True])
def test_score_numdiff(noise):
    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(1000, model1, noise)
    preg = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)

    def loglike(x):
        return preg.loglike(x)
    q = x_mean.shape[1] + x_sc.shape[1] + x_sm.shape[1]
    if noise:
        q += x_no.shape[1]
    np.random.seed(342)
    atol = 0.002 if PLATFORM_OSX else 0.01
    for _ in range(5):
        par0 = preg._get_start()
        par = par0 + 0.1 * np.random.normal(size=q)
        score = preg.score(par)
        score_nd = nd.approx_fprime(par, loglike, epsilon=1e-07)
        assert_allclose(score, score_nd, atol=atol, rtol=0.0001)