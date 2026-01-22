import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
class ConstrainedCompareWtdMixin(ConstrainedCompareMixin):

    @classmethod
    def setup_class(cls):
        nobs, k_exog = (100, 5)
        np.random.seed(987125)
        x = np.random.randn(nobs, k_exog - 1)
        x = add_constant(x)
        cls.aweights = np.random.randint(1, 10, nobs)
        y_true = x.sum(1) / 2
        y = y_true + 2 * np.random.randn(nobs)
        cls.endog = y
        cls.exog = x
        cls.idx_uc = [0, 2, 3, 4]
        cls.idx_p_uc = np.array(cls.idx_uc)
        cls.idx_c = [1]
        cls.exogc = xc = x[:, cls.idx_uc]
        mod_ols_c = WLS(y - 0.5 * x[:, 1], xc, weights=cls.aweights)
        mod_ols_c.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.mod2 = mod_ols_c
        cls.init()