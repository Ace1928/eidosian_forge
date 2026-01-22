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
class TestProbitL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 4
        cls.m = 3
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=True)
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = Probit(data.endog, data.exog).fit_regularized(method='l1', alpha=alpha, disp=0, acc=1e-15, maxiter=2000, trim_mode='auto')
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = Probit(data.endog, exog_no_PSI).fit(disp=0, tol=1e-15)