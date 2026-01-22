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
class TestLogitNewtonPrepend(CheckMargEff):

    @classmethod
    def setup_class(cls):
        data = load_spector()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.res1 = Logit(data.endog, data.exog).fit(method='newton', disp=0)
        res2 = Spector.logit
        cls.res2 = res2
        cls.slice = np.roll(np.arange(len(cls.res1.params)), 1)

    def test_resid_pearson(self):
        assert_almost_equal(self.res1.resid_pearson, self.res2.resid_pearson, 5)

    def test_nodummy_exog1(self):
        me = self.res1.get_margeff(atexog={1: 2.0, 3: 1.0})
        assert_almost_equal(me.margeff, self.res2.margeff_nodummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se, self.res2.margeff_nodummy_atexog1_se, DECIMAL_4)

    def test_nodummy_exog2(self):
        me = self.res1.get_margeff(atexog={2: 21.0, 3: 0}, at='mean')
        assert_almost_equal(me.margeff, self.res2.margeff_nodummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se, self.res2.margeff_nodummy_atexog2_se, DECIMAL_4)

    def test_dummy_exog1(self):
        me = self.res1.get_margeff(atexog={1: 2.0, 3: 1.0}, dummy=True)
        assert_almost_equal(me.margeff, self.res2.margeff_dummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dummy_atexog1_se, DECIMAL_4)

    def test_dummy_exog2(self):
        me = self.res1.get_margeff(atexog={2: 21.0, 3: 0}, at='mean', dummy=True)
        assert_almost_equal(me.margeff, self.res2.margeff_dummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dummy_atexog2_se, DECIMAL_4)