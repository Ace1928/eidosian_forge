import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_poisson_2d():
    y = np.r_[3, 1, 4, 8, 2, 5, 4, 7, 2, 6]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[2, 1, 0, 0, 1, 2, 3, 2, 0, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2
    model = ConditionalPoisson(y, x, groups=g)
    for x in (-1, 0, 1, 2):
        params = np.r_[-0.5 * x, 0.5 * x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=0.0001)
    result = model.fit()
    assert_allclose(result.params, np.r_[-0.9478957, -0.0134279], rtol=0.001)
    assert_allclose(result.bse, np.r_[0.3874942, 0.1686712], rtol=1e-05)
    result.summary()