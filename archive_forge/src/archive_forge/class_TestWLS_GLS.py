from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestWLS_GLS(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.ccard import load
        data = load()
        endog = np.asarray(data.endog)
        exog = np.asarray(data.exog)
        sigma = exog[:, 2]
        cls.res1 = WLS(endog, exog, weights=1 / sigma).fit()
        cls.res2 = GLS(endog, exog, sigma=sigma).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL_4)