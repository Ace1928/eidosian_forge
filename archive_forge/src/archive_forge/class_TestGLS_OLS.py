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
class TestGLS_OLS(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        data = longley.load()
        endog = np.asarray(data.endog)
        exog = np.asarray(data.exog)
        exog = add_constant(exog, prepend=False)
        cls.res1 = GLS(endog, exog).fit()
        cls.res2 = OLS(endog, exog).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL_4)