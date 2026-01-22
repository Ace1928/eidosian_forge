import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
class TestGaussianLog(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_aic_Stata = DECIMAL_2
        cls.decimal_loglike = DECIMAL_0
        cls.decimal_null_deviance = DECIMAL_1
        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
        cls.X = np.c_[np.ones((nobs, 1)), x, x ** 2]
        cls.lny = np.exp(-(-1.0 + 0.02 * x + 0.0001 * x ** 2)) + 0.001 * np.random.randn(nobs)
        GaussLog_Model = GLM(cls.lny, cls.X, family=sm.families.Gaussian(sm.families.links.Log()))
        cls.res1 = GaussLog_Model.fit()
        from .results.results_glm import GaussianLog
        cls.res2 = GaussianLog()