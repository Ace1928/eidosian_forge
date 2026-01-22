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
class TestTweediePower2(CheckTweedie):

    @classmethod
    def setup_class(cls):
        from .results.results_glm import CpunishTweediePower2
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.Power(1), var_power=2.0)
        cls.res1 = sm.GLM(endog=cls.data.endog, exog=cls.data.exog[['INCOME', 'SOUTH']], family=family_link).fit()
        cls.res2 = CpunishTweediePower2()