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
class TestWtdGlmGammaNewton(CheckWtdDuplicationMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Gamma family with log link.
        """
        super().setup_class()
        family_link = sm.families.Gamma(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog, freq_weights=cls.weight, family=family_link).fit(method='newton')
        cls.res2 = GLM(cls.endog_big, cls.exog_big, family=family_link).fit(method='newton')

    def test_init_kwargs(self):
        family_link = sm.families.Gamma(sm.families.links.Log())
        with pytest.warns(ValueWarning, match='unknown kwargs'):
            GLM(self.endog, self.exog, family=family_link, weights=self.weight)