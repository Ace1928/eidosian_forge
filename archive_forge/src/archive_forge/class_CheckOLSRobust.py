import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
class CheckOLSRobust:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        assert_allclose(res1.params, res2.params, rtol=rtol)
        assert_allclose(self.bse_robust, res2.bse, rtol=rtol)
        assert_allclose(self.cov_robust, res2.cov, rtol=rtol)

    @pytest.mark.smoke
    def test_t_test_summary(self):
        res1 = self.res1
        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        tt.summary()

    @pytest.mark.smoke
    def test_t_test_summary_frame(self):
        res1 = self.res1
        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        tt.summary_frame()

    @pytest.mark.smoke
    def test_f_test_summary(self):
        res1 = self.res1
        mat = np.eye(len(res1.params))
        ft = res1.f_test(mat[:-1], cov_p=self.cov_robust)
        ft.summary()

    def test_tests(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        rtolh = getattr(self, 'rtolh', 1e-12)
        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        assert_allclose(tt.effect, res2.params, rtol=rtol)
        assert_allclose(tt.sd, res2.bse, rtol=rtol)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=rtol)
        if self.small:
            assert_allclose(tt.pvalue, res2.pvalues, rtol=5 * rtol)
        else:
            pval = stats.norm.sf(np.abs(tt.tvalue)) * 2
            assert_allclose(pval, res2.pvalues, rtol=5 * rtol, atol=1e-25)
        ft = res1.f_test(mat[:-1], cov_p=self.cov_robust)
        if self.small:
            assert_allclose(ft.fvalue, res2.F, rtol=rtol)
            if hasattr(res2, 'Fp'):
                assert_allclose(ft.pvalue, res2.Fp, rtol=rtol)
        elif not getattr(self, 'skip_f', False):
            dof_corr = res1.df_resid * 1.0 / res1.nobs
            assert_allclose(ft.fvalue * dof_corr, res2.F, rtol=rtol)
        if hasattr(res2, 'df_r'):
            assert_equal(ft.df_num, res2.df_m)
            assert_equal(ft.df_denom, res2.df_r)
        else:
            assert_equal(ft.df_num, res2.Fdf1)
            assert_equal(ft.df_denom, res2.Fdf2)