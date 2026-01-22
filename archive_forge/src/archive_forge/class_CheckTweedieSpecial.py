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
class CheckTweedieSpecial:

    def test_mu(self):
        assert_allclose(self.res1.mu, self.res2.mu, rtol=1e-05, atol=1e-05)

    def test_resid(self):
        assert_allclose(self.res1.resid_response, self.res2.resid_response, rtol=1e-05, atol=1e-05)
        assert_allclose(self.res1.resid_pearson, self.res2.resid_pearson, rtol=1e-05, atol=1e-05)
        assert_allclose(self.res1.resid_deviance, self.res2.resid_deviance, rtol=1e-05, atol=1e-05)
        assert_allclose(self.res1.resid_working, self.res2.resid_working, rtol=1e-05, atol=1e-05)
        assert_allclose(self.res1.resid_anscombe_unscaled, self.res2.resid_anscombe_unscaled, rtol=1e-05, atol=1e-05)