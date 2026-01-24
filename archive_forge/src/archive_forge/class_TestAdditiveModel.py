from statsmodels.compat.python import lrange
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats
import pytest
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
class TestAdditiveModel(BaseAM, CheckAM):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        nobs = cls.nobs
        y_true, x, exog = (cls.y_true, cls.x, cls.exog)
        np.random.seed(8928993)
        sigma_noise = 0.1
        y = y_true + sigma_noise * np.random.randn(nobs)
        m = AdditiveModel(x)
        m.fit(y)
        res_gam = m.results
        res_ols = OLS(y, exog).fit()
        cls.res1 = res1 = Dummy()
        cls.res2 = res2 = res_ols
        res1.y_pred = res_gam.predict(x)
        res2.y_pred = res_ols.model.predict(res_ols.params, exog)
        res1.y_predshort = res_gam.predict(x[:10])
        slopes = [i for ss in m.smoothers for i in ss.params[1:]]
        const = res_gam.alpha + sum([ss.params[1] for ss in m.smoothers])
        res1.params = np.array([const] + slopes)

    def test_fitted(self):
        super().test_fitted()