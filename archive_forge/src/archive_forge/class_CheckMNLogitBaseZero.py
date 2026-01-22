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
class CheckMNLogitBaseZero(CheckModelResults):

    def test_margeff_overall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_overall, 6)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_overall_se, 6)
        me_frame = me.summary_frame()
        eff = me_frame['dy/dx'].values.reshape(me.margeff.shape, order='F')
        assert_allclose(eff, me.margeff, rtol=1e-13)
        assert_equal(me_frame.shape, (np.size(me.margeff), 6))

    def test_margeff_mean(self):
        me = self.res1.get_margeff(at='mean')
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_mean, 7)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_mean_se, 7)

    def test_margeff_dummy(self):
        data = self.data
        vote = data.data['vote']
        exog = np.column_stack((data.exog, vote))
        exog = sm.add_constant(exog, prepend=False)
        res = MNLogit(data.endog, exog).fit(method='newton', disp=0)
        me = res.get_margeff(dummy=True)
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_dummy_overall, 6)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_dummy_overall_se, 6)
        me = res.get_margeff(dummy=True, method='eydx')
        assert_almost_equal(me.margeff, self.res2.margeff_eydx_dummy_overall, 5)
        assert_almost_equal(me.margeff_se, self.res2.margeff_eydx_dummy_overall_se, 6)

    def test_j(self):
        assert_equal(self.res1.model.J, self.res2.J)

    def test_k(self):
        assert_equal(self.res1.model.K, self.res2.K)

    def test_endog_names(self):
        assert_equal(self.res1._get_endog_name(None, None)[1], ['y=1', 'y=2', 'y=3', 'y=4', 'y=5', 'y=6'])

    def test_pred_table(self):
        pred = [6, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 6, 0, 1, 6, 0, 0, 1, 1, 6, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 6, 0, 0, 6, 6, 0, 0, 1, 1, 6, 1, 6, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6, 0, 0, 6, 0, 0, 0, 1, 1, 0, 0, 6, 6, 6, 6, 1, 0, 5, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 6, 0, 6, 6, 1, 0, 1, 1, 6, 5, 1, 0, 0, 0, 5, 0, 0, 6, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 6, 6, 6, 6, 5, 0, 1, 1, 0, 1, 0, 6, 6, 0, 0, 0, 6, 0, 0, 0, 6, 6, 0, 5, 1, 0, 0, 0, 0, 6, 0, 5, 6, 6, 0, 0, 0, 0, 6, 1, 0, 0, 1, 0, 1, 6, 1, 1, 1, 1, 1, 0, 0, 0, 6, 0, 5, 1, 0, 6, 6, 6, 0, 0, 0, 0, 1, 6, 6, 0, 0, 0, 1, 1, 5, 6, 0, 6, 1, 0, 0, 1, 6, 0, 0, 1, 0, 6, 6, 0, 5, 6, 6, 0, 0, 6, 1, 0, 6, 0, 1, 0, 1, 6, 0, 1, 1, 1, 6, 0, 5, 0, 0, 6, 1, 0, 6, 5, 5, 0, 6, 1, 1, 1, 0, 0, 6, 0, 0, 5, 0, 0, 6, 6, 6, 6, 6, 0, 1, 0, 0, 6, 6, 0, 0, 1, 6, 0, 0, 6, 1, 6, 1, 1, 1, 0, 1, 6, 5, 0, 0, 1, 5, 0, 1, 6, 6, 1, 0, 0, 1, 6, 1, 5, 6, 1, 0, 0, 1, 1, 0, 6, 1, 6, 0, 1, 1, 5, 6, 6, 5, 1, 1, 1, 0, 6, 1, 6, 1, 0, 1, 0, 0, 1, 5, 0, 1, 1, 0, 5, 6, 0, 5, 1, 1, 6, 5, 0, 6, 0, 0, 0, 0, 0, 0, 1, 6, 1, 0, 5, 1, 0, 0, 1, 6, 0, 0, 6, 6, 6, 0, 2, 1, 6, 5, 6, 1, 1, 0, 5, 1, 1, 1, 6, 1, 6, 6, 5, 6, 0, 1, 0, 1, 6, 0, 6, 1, 6, 0, 0, 6, 1, 0, 6, 1, 0, 0, 0, 0, 6, 6, 6, 6, 5, 6, 6, 0, 0, 6, 1, 1, 6, 0, 0, 6, 6, 0, 6, 6, 0, 0, 6, 0, 0, 6, 6, 6, 1, 0, 6, 0, 0, 0, 6, 1, 1, 0, 1, 5, 0, 0, 5, 0, 0, 0, 1, 1, 6, 1, 0, 0, 0, 6, 6, 1, 1, 6, 5, 5, 0, 6, 6, 0, 1, 1, 0, 6, 6, 0, 6, 5, 5, 6, 5, 1, 0, 6, 0, 6, 1, 0, 1, 6, 6, 6, 1, 0, 6, 0, 5, 6, 6, 5, 0, 5, 1, 0, 6, 0, 6, 1, 5, 5, 0, 1, 5, 5, 2, 6, 6, 6, 5, 0, 0, 1, 6, 1, 0, 1, 6, 1, 0, 0, 1, 5, 6, 6, 0, 0, 0, 5, 6, 6, 6, 1, 5, 6, 1, 0, 0, 6, 5, 0, 1, 1, 1, 6, 6, 0, 1, 0, 0, 0, 5, 0, 0, 6, 1, 6, 0, 6, 1, 5, 5, 6, 5, 0, 0, 0, 0, 1, 1, 0, 5, 5, 0, 0, 0, 0, 1, 0, 6, 6, 1, 1, 6, 6, 0, 5, 5, 0, 0, 0, 6, 6, 1, 6, 0, 0, 5, 0, 1, 6, 5, 6, 6, 5, 5, 6, 6, 1, 0, 1, 6, 6, 1, 6, 0, 6, 0, 6, 5, 0, 6, 6, 0, 5, 6, 0, 6, 6, 5, 0, 1, 6, 6, 1, 0, 1, 0, 6, 6, 1, 0, 6, 6, 6, 0, 1, 6, 0, 1, 5, 1, 1, 5, 6, 6, 0, 1, 6, 6, 1, 5, 0, 5, 0, 6, 0, 1, 6, 1, 0, 6, 1, 6, 0, 6, 1, 0, 0, 0, 6, 6, 0, 1, 1, 6, 6, 6, 1, 6, 0, 5, 6, 0, 5, 6, 6, 5, 5, 5, 6, 0, 6, 0, 0, 0, 5, 0, 6, 1, 2, 6, 6, 6, 5, 1, 6, 0, 6, 0, 0, 0, 0, 6, 5, 0, 5, 1, 6, 5, 1, 6, 5, 1, 1, 0, 0, 6, 1, 1, 5, 6, 6, 0, 5, 2, 5, 5, 0, 5, 5, 5, 6, 5, 6, 6, 5, 2, 6, 5, 6, 0, 0, 6, 5, 0, 6, 0, 0, 6, 6, 6, 0, 5, 1, 1, 6, 6, 5, 2, 1, 6, 5, 6, 0, 6, 6, 1, 1, 5, 1, 6, 6, 6, 0, 0, 6, 1, 0, 5, 5, 1, 5, 6, 1, 6, 0, 1, 6, 5, 0, 0, 6, 1, 5, 1, 0, 6, 0, 6, 6, 5, 5, 6, 6, 6, 6, 2, 6, 6, 6, 5, 5, 5, 0, 1, 0, 0, 0, 6, 6, 1, 0, 6, 6, 6, 6, 6, 1, 0, 6, 1, 5, 5, 6, 6, 6, 6, 6, 5, 6, 1, 6, 2, 5, 5, 6, 5, 6, 6, 5, 6, 6, 5, 5, 6, 1, 5, 1, 6, 0, 2, 5, 0, 5, 0, 2, 1, 6, 0, 0, 6, 6, 1, 6, 0, 5, 5, 6, 6, 1, 6, 6, 6, 5, 6, 6, 1, 6, 5, 6, 1, 1, 0, 6, 6, 5, 1, 0, 0, 6, 6, 5, 6, 0, 1, 6, 0, 5, 6, 5, 2, 5, 2, 0, 0, 1, 6, 6, 1, 5, 6, 6, 0, 6, 6, 6, 6, 6, 5]
        assert_array_equal(self.res1.predict().argmax(1), pred)
        assert_array_equal(self.res1.pred_table().sum(0), np.bincount(pred))
        pred = [[126.0, 41.0, 2.0, 0.0, 0.0, 12.0, 19.0], [77.0, 73.0, 3.0, 0.0, 0.0, 15.0, 12.0], [37.0, 43.0, 2.0, 0.0, 0.0, 19.0, 7.0], [12.0, 9.0, 1.0, 0.0, 0.0, 9.0, 6.0], [19.0, 10.0, 2.0, 0.0, 0.0, 20.0, 43.0], [22.0, 25.0, 1.0, 0.0, 0.0, 31.0, 71.0], [9.0, 7.0, 1.0, 0.0, 0.0, 18.0, 140.0]]
        assert_array_equal(self.res1.pred_table(), pred)

    def test_resid(self):
        assert_array_equal(self.res1.resid_misclassified, self.res2.resid)

    @pytest.mark.xfail(reason='res2.cov_params is a zero-dim array of None', strict=True)
    def test_cov_params(self):
        super().test_cov_params()

    @pytest.mark.xfail(reason='Test has not been implemented for this class.', strict=True, raises=NotImplementedError)
    def test_distr(self):
        super().test_distr()