import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning
class TestPoissonMLE(CompareMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(98928678)
        nobs = 200
        rvs = np.random.randn(nobs, 6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog, prepend=False)
        xbeta = 0.1 + 0.1 * rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))
        cls.res_discrete = Poisson(data_endog, data_exog).fit(disp=0)
        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()
        cls.mod = PoissonGMLE(data_endog, data_exog)
        cls.res = cls.mod.fit(start_params=0.9 * cls.res_discrete.params, method='bfgs', disp=0)

    def test_predict_distribution(self):
        res = self.res
        model = self.mod
        with pytest.raises(ValueError):
            model.predict_distribution(model.exog)
        try:
            model.result = res
            dist = model.predict_distribution(model.exog)
            assert isinstance(dist, stats._distn_infrastructure.rv_frozen)
            assert_almost_equal(dist.mean(), np.exp(model.exog.dot(res.params)), 15)
        finally:
            model.__delattr__('result')