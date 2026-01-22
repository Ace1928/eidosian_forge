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
class TestRTO(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        from .results.results_regression import LongleyRTO
        data = longley.load()
        endog = np.asarray(data.endog)
        exog = np.asarray(data.exog)
        res1 = OLS(endog, exog).fit()
        res2 = LongleyRTO()
        res2.wresid = res1.wresid
        cls.res1 = res1
        cls.res2 = res2
        res_qr = OLS(endog, exog).fit(method='qr')
        cls.res_qr = res_qr