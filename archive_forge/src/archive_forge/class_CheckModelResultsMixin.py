import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
class CheckModelResultsMixin:

    def test_params(self):
        assert_allclose(np.ravel(self.res1.params.loc[idx]), self.res2.table[:, 0], rtol=0.001)

    def test_bse(self):
        assert_equal(self.res1.scale, 1)
        assert_allclose(np.ravel(self.res1.bse.loc[idx]), self.res2.table[:, 1], rtol=0.001)

    def test_tvalues(self):
        assert_allclose(np.ravel(self.res1.tvalues.loc[idx]), self.res2.table[:, 2], rtol=0.01)

    def test_pvalues(self):
        pvals_stata = scipy.stats.t.sf(self.res2.table[:, 2], self.res2.df_r)
        assert_allclose(np.ravel(self.res1.pvalues.loc[idx]), pvals_stata, rtol=1.1)
        pvals_t = scipy.stats.t.sf(self.res1.tvalues, self.res2.df_r) * 2
        assert_allclose(np.ravel(self.res1.pvalues), pvals_t, rtol=1e-09, atol=1e-10)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int().loc[idx], self.res2.table[:, -2:], rtol=0.001)

    def test_nobs(self):
        assert_allclose(self.res1.nobs, self.res2.N, rtol=0.001)

    def test_df_model(self):
        assert_allclose(self.res1.df_model, self.res2.df_m, rtol=0.001)

    def test_df_resid(self):
        assert_allclose(self.res1.df_resid, self.res2.df_r, rtol=0.001)

    def test_prsquared(self):
        assert_allclose(self.res1.prsquared, self.res2.psrsquared, rtol=0.001)

    def test_sparsity(self):
        assert_allclose(np.array(self.res1.sparsity), self.res2.sparsity, rtol=0.001)

    def test_bandwidth(self):
        assert_allclose(np.array(self.res1.bandwidth), self.res2.kbwidth, rtol=0.001)