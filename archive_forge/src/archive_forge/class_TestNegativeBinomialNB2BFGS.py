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
class TestNegativeBinomialNB2BFGS(CheckNegBinMixin, CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomial(data.endog, exog, 'nb2').fit(method='bfgs', disp=0, maxiter=1000)
        res2 = RandHIE.negativebinomial_nb2_bfgs
        cls.res2 = res2

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_3)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_alpha(self):
        self.res1.bse
        assert_almost_equal(self.res1.lnalpha, self.res2.lnalpha, DECIMAL_4)
        assert_almost_equal(self.res1.lnalpha_std_err, self.res2.lnalpha_std_err, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL_3)

    def test_zstat(self):
        assert_almost_equal(self.res1.pvalues[:-1], self.res2.pvalues, DECIMAL_2)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues[:10], self.res2.fittedvalues[:10], DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.predict()[:10], np.exp(self.res2.fittedvalues[:10]), DECIMAL_3)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.predict(which='linear')[:10], self.res2.fittedvalues[:10], DECIMAL_3)