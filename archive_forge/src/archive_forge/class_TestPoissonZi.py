import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning
class TestPoissonZi(CompareMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(98928678)
        nobs = 200
        rvs = np.random.randn(nobs, 6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog, prepend=False)
        xbeta = 1 + 0.1 * rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))
        cls.k_extra = 1
        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()
        cls.res_discrete = Poisson(data_endog, data_exog).fit(disp=0)
        offset = cls.res_discrete.params[0] * data_exog[:, 0]
        cls.res_discrete = Poisson(data_endog, data_exog[:, 1:], offset=offset).fit(disp=0)
        cls.res = PoissonZiGMLE(data_endog, data_exog[:, 1:], offset=offset).fit(start_params=np.r_[0.9 * cls.res_discrete.params, 10], method='bfgs', disp=0)
        cls.decimal = 4

    def test_params(self):
        assert_almost_equal(self.res.params[:-1], self.res_glm.params[1:], self.decimal)
        assert_almost_equal(self.res.params[:-1], self.res_discrete.params, self.decimal)

    def test_cov_params(self):
        assert_almost_equal(self.res.bsejac[:-1], self.res_glm.bse[1:], self.decimal - 2)
        assert_almost_equal(self.res.bsejac[:-1], self.res_discrete.bse, self.decimal - 2)

    def test_exog_names_warning(self):
        mod = self.res.model
        mod1 = PoissonOffsetGMLE(mod.endog, mod.exog, offset=mod.offset)
        from numpy.testing import assert_warns
        mod1.data.xnames = mod1.data.xnames * 2
        assert_warns(ValueWarning, mod1.fit, disp=0)