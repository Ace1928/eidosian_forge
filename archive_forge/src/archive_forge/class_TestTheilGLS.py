import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class TestTheilGLS(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        nobs = len(y)
        weights = (np.arange(nobs) < nobs // 2) + 0.5
        mod1 = TheilGLS(y, x, sigma=weights, sigma_prior=[0, 0, 1.0, 1.0])
        cls.res1 = mod1.fit(200000)
        cls.res2 = GLS(y, x[:, :3], sigma=weights).fit()