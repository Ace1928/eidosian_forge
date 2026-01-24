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
class BaseGAM(BaseAM, CheckGAM):

    @classmethod
    def init(cls):
        nobs = cls.nobs
        y_true, x, exog = (cls.y_true, cls.x, cls.exog)
        if not hasattr(cls, 'scale'):
            scale = 1
        else:
            scale = cls.scale
        f = cls.family
        cls.mu_true = mu_true = f.link.inverse(y_true)
        np.random.seed(8928993)
        try:
            y_obs = cls.rvs(mu_true, scale=scale, size=nobs)
        except TypeError:
            y_obs = cls.rvs(mu_true, size=nobs)
        m = GAM(y_obs, x, family=f)
        m.fit(y_obs, maxiter=100)
        res_gam = m.results
        cls.res_gam = res_gam
        cls.mod_gam = m
        res_glm = GLM(y_obs, exog, family=f).fit()
        cls.res1 = res1 = Dummy()
        cls.res2 = res2 = res_glm
        res2.y_pred = res_glm.model.predict(res_glm.params, exog, which='linear')
        res1.y_pred = res_gam.predict(x)
        res1.y_predshort = res_gam.predict(x[:10])
        res2.mu_pred = res_glm.model.predict(res_glm.params, exog, which='mean')
        res1.mu_pred = res_gam.mu
        slopes = [i for ss in m.smoothers for i in ss.params[1:]]
        const = res_gam.alpha + sum([ss.params[1] for ss in m.smoothers])
        res1.params = np.array([const] + slopes)