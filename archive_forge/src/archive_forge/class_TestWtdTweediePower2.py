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
class TestWtdTweediePower2(CheckWtdDuplicationMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Tweedie family with Power(1) link and var_power=2.
        """
        cls.data = cpunish.load_pandas()
        cls.endog = cls.data.endog
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        np.random.seed(1234)
        cls.weight = np.random.randint(5, 100, len(cls.endog))
        cls.endog_big = np.repeat(cls.endog.values, cls.weight)
        cls.exog_big = np.repeat(cls.exog.values, cls.weight, axis=0)
        link = sm.families.links.Power()
        family_link = sm.families.Tweedie(link=link, var_power=2)
        cls.res1 = GLM(cls.endog, cls.exog, freq_weights=cls.weight, family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big, family=family_link).fit()