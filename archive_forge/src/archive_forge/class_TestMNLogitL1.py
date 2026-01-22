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
class TestMNLogitL1(CheckLikelihoodModelL1):

    @classmethod
    def setup_class(cls):
        anes_data = load_anes96()
        anes_exog = anes_data.exog
        anes_exog = sm.add_constant(anes_exog, prepend=False)
        mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
        alpha = 10.0 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
        alpha[-1, :] = 0
        cls.res1 = mlogit_mod.fit_regularized(method='l1', alpha=alpha, trim_mode='auto', auto_trim_tol=0.02, acc=1e-10, disp=0)
        res2 = DiscreteL1.mnlogit
        cls.res2 = res2