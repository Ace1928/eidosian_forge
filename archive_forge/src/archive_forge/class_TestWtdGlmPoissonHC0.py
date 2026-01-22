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
class TestWtdGlmPoissonHC0(CheckWtdDuplicationMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Poisson family with canonical log link.
        """
        super().setup_class()
        start_params = np.array([0.000182794424, -0.0476785037, -0.0948249717, -0.000292293226, 2.63728909, -20.5934384])
        fit_kwds = dict(cov_type='HC0')
        cls.res1 = GLM(cls.endog, cls.exog, freq_weights=cls.weight, family=sm.families.Poisson()).fit(**fit_kwds)
        fit_kwds = dict(cov_type='HC0', start_params=start_params)
        cls.res2 = GLM(cls.endog_big, cls.exog_big, family=sm.families.Poisson()).fit(**fit_kwds)