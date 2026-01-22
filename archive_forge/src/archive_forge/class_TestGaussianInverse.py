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
class TestGaussianInverse(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        cls.decimal_bic = DECIMAL_1
        cls.decimal_aic_R = DECIMAL_1
        cls.decimal_aic_Stata = DECIMAL_3
        cls.decimal_loglike = DECIMAL_1
        cls.decimal_resids = DECIMAL_3
        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
        y = 1.0 + 2.0 * x + x ** 2 + 0.1 * np.random.randn(nobs)
        cls.X = np.c_[np.ones((nobs, 1)), x, x ** 2]
        cls.y_inv = (1.0 + 0.02 * x + 0.001 * x ** 2) ** (-1) + 0.001 * np.random.randn(nobs)
        InverseLink_Model = GLM(cls.y_inv, cls.X, family=sm.families.Gaussian(sm.families.links.InversePower()))
        InverseLink_Res = InverseLink_Model.fit()
        cls.res1 = InverseLink_Res
        from .results.results_glm import GaussianInverse
        cls.res2 = GaussianInverse()