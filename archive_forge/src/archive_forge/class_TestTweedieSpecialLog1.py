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
class TestTweedieSpecialLog1(CheckTweedieSpecial):

    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family1 = sm.families.Poisson(link=sm.families.links.Log())
        cls.res1 = sm.GLM(endog=cls.data.endog, exog=cls.data.exog[['INCOME', 'SOUTH']], family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.Log(), var_power=1)
        cls.res2 = sm.GLM(endog=cls.data.endog, exog=cls.data.exog[['INCOME', 'SOUTH']], family=family2).fit()