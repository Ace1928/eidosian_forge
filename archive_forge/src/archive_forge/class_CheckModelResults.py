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
class CheckModelResults(CheckModelMixin):
    """
    res2 should be the test results from RModelWrap
    or the results as defined in model_results_data
    """

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, rtol=8e-05)

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_4)

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

    def test_cov_params(self):
        if not hasattr(self.res2, 'cov_params'):
            raise pytest.skip('TODO: implement res2.cov_params')
        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params, DECIMAL_4)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)

    def test_llnull(self):
        assert_almost_equal(self.res1.llnull, self.res2.llnull, DECIMAL_4)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL_3)

    def test_llr_pvalue(self):
        assert_almost_equal(self.res1.llr_pvalue, self.res2.llr_pvalue, DECIMAL_4)

    @pytest.mark.xfail(reason='Test has not been implemented for this class.', strict=True, raises=NotImplementedError)
    def test_normalized_cov_params(self):
        raise NotImplementedError

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_dof(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.model.predict(self.res1.params), self.res2.phat, DECIMAL_4)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.model.predict(self.res1.params, which='linear'), self.res2.yhat, DECIMAL_4)

    def test_loglikeobs(self):
        llobssum = self.res1.model.loglikeobs(self.res1.params).sum()
        assert_almost_equal(llobssum, self.res1.llf, DECIMAL_14)

    def test_jac(self):
        check_jac(self)

    def test_summary_latex(self):
        summ = self.res1.summary()
        ltx = summ.as_latex()
        n_lines = len(ltx.splitlines())
        if not isinstance(self.res1.model, MNLogit):
            assert n_lines == 19 + np.size(self.res1.params)
        assert 'Covariance Type:' in ltx

    def test_distr(self):
        check_distr(self.res1)