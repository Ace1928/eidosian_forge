import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS
class BasePolySmoother:

    @classmethod
    def setup_class(cls):
        order = 3
        sigma_noise = 0.5
        nobs = 100
        lb, ub = (-1, 2)
        cls.x = x = np.linspace(lb, ub, nobs)
        cls.exog = exog = x[:, None] ** np.arange(order + 1)
        y_true = exog.sum(1)
        np.random.seed(987567)
        cls.y = y = y_true + sigma_noise * np.random.randn(nobs)