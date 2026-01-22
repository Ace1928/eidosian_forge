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
class TestGlmGamma(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Gamma family with canonical inverse link (power -1)
        """
        cls.decimal_aic_R = -1
        cls.decimal_resids = DECIMAL_2
        from statsmodels.datasets.scotland import load
        from .results.results_glm import Scotvote
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res1 = GLM(data.endog, data.exog, family=sm.families.Gamma()).fit()
        cls.res1 = res1
        res2 = Scotvote()
        res2.aic_R += 2
        cls.res2 = res2