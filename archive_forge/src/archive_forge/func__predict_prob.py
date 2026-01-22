import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
def _predict_prob(self, params, exog, exog_infl, exposure, offset, y_values=None):
    params_infl = params[:self.k_inflate]
    params_main = params[self.k_inflate:]
    p = self.model_main.parameterization
    if y_values is None:
        y_values = np.arange(0, np.max(self.endog) + 1)
    if len(exog_infl.shape) < 2:
        transform = True
        w = np.atleast_2d(self.model_infl.predict(params_infl, exog_infl))[:, None]
    else:
        transform = False
        w = self.model_infl.predict(params_infl, exog_infl)[:, None]
    w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
    mu = self.model_main.predict(params_main, exog, exposure=exposure, offset=offset)[:, None]
    result = self.distribution.pmf(y_values, mu, params_main[-1], p, w)
    return result[0] if transform else result