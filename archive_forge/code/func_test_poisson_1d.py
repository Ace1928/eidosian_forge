import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_poisson_1d():
    y = np.r_[3, 1, 1, 4, 5, 2, 0, 1, 6, 2]
    g = np.r_[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]
    model = ConditionalPoisson(y, x, groups=g)
    for x in (-1, 0, 1, 2):
        grad = approx_fprime(np.r_[x,], model.loglike).squeeze()
        score = model.score(np.r_[x,])
        assert_allclose(grad, score, rtol=0.0001)
    result = model.fit()
    assert_allclose(result.params, np.r_[0.6466272], rtol=0.0001)
    assert_allclose(result.bse, np.r_[0.4170918], rtol=1e-05)