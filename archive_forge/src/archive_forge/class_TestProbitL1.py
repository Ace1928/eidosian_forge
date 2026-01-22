from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class TestProbitL1(CheckLikelihoodModelL1):

    @classmethod
    def setup_class(cls):
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=True)
        alpha = np.array([0.1, 0.2, 0.3, 10])
        cls.res1 = Probit(data.endog, data.exog).fit_regularized(method='l1', alpha=alpha, disp=0, trim_mode='auto', auto_trim_tol=0.02, acc=1e-10, maxiter=1000)
        res2 = DiscreteL1.probit
        cls.res2 = res2

    def test_cov_params(self):
        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params, DECIMAL_4)