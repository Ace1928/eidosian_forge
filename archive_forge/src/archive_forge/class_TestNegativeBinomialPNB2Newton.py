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
class TestNegativeBinomialPNB2Newton(CheckNegBinMixin, CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        exog = sm.add_constant(data.exog, prepend=False)
        mod = NegativeBinomialP(data.endog, exog, p=2)
        cls.res1 = mod.fit(method='newton', disp=0)
        res2 = RandHIE.negativebinomial_nb2_bfgs
        cls.res2 = res2

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=0.001, rtol=0.001)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-07)

    def test_alpha(self):
        self.res1.bse
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha)
        assert_allclose(self.res1.lnalpha_std_err, self.res2.lnalpha_std_err, atol=1e-07)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=0.001, rtol=0.001)

    def test_zstat(self):
        assert_allclose(self.res1.pvalues[:-1], self.res2.pvalues, atol=0.005, rtol=0.005)

    def test_fittedvalues(self):
        assert_allclose(self.res1.fittedvalues[:10], self.res2.fittedvalues[:10])

    def test_predict(self):
        assert_allclose(self.res1.predict()[:10], np.exp(self.res2.fittedvalues[:10]))

    def test_predict_xb(self):
        assert_allclose(self.res1.predict(which='linear')[:10], self.res2.fittedvalues[:10])