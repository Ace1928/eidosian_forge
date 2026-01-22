import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
class ConstrainedCompareMixin:

    @classmethod
    def setup_class(cls):
        nobs, k_exog = (100, 5)
        np.random.seed(987125)
        x = np.random.randn(nobs, k_exog - 1)
        x = add_constant(x)
        y_true = x.sum(1) / 2
        y = y_true + 2 * np.random.randn(nobs)
        cls.endog = y
        cls.exog = x
        cls.idx_uc = [0, 2, 3, 4]
        cls.idx_p_uc = np.array(cls.idx_uc)
        cls.idx_c = [1]
        cls.exogc = xc = x[:, cls.idx_uc]
        mod_ols_c = OLS(y - 0.5 * x[:, 1], xc)
        mod_ols_c.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.mod2 = mod_ols_c
        cls.init()

    def test_params(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params[self.idx_p_uc], res2.params, rtol=1e-10)

    def test_se(self):
        res1 = self.res1
        res2 = self.res2
        assert_equal(res1.df_resid, res2.df_resid)
        assert_allclose(res1.scale, res2.scale, rtol=1e-10)
        assert_allclose(res1.bse[self.idx_p_uc], res2.bse, rtol=1e-10)
        assert_allclose(res1.cov_params()[self.idx_p_uc[:, None], self.idx_p_uc], res2.cov_params(), rtol=5e-09, atol=1e-15)

    def test_resid(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.resid_response, res2.resid, rtol=1e-10)