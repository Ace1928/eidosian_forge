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
def _hessian_main(self, params):
    params_infl = params[:self.k_inflate]
    params_main = params[self.k_inflate:]
    y = self.endog
    w = self.model_infl.predict(params_infl)
    w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
    score = self.score(params)
    zero_idx = np.nonzero(y == 0)[0]
    nonzero_idx = np.nonzero(y)[0]
    mu = self.model_main.predict(params_main)
    hess_arr = np.zeros((self.k_exog, self.k_exog))
    coeff = 1 + w[zero_idx] * (np.exp(mu[zero_idx]) - 1)
    for i in range(self.k_exog):
        for j in range(i, -1, -1):
            hess_arr[i, j] = (self.exog[zero_idx, i] * self.exog[zero_idx, j] * mu[zero_idx] * (w[zero_idx] - 1) * (1 / coeff - w[zero_idx] * mu[zero_idx] * np.exp(mu[zero_idx]) / coeff ** 2)).sum() - (mu[nonzero_idx] * self.exog[nonzero_idx, i] * self.exog[nonzero_idx, j]).sum()
    return hess_arr