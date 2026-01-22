from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
def _hessian_nb2(self, params):
    """
        Hessian of NB2 model.
        """
    if self._transparams:
        alpha = np.exp(params[-1])
    else:
        alpha = params[-1]
    a1 = 1 / alpha
    params = params[:-1]
    exog = self.exog
    y = self.endog[:, None]
    mu = self.predict(params)[:, None]
    prob = a1 / (a1 + mu)
    dgpart = digamma(a1 + y) - digamma(a1)
    dim = exog.shape[1]
    hess_arr = np.empty((dim + 1, dim + 1))
    const_arr = a1 * mu * (a1 + y) / (mu + a1) ** 2
    for i in range(dim):
        for j in range(dim):
            if j > i:
                continue
            hess_arr[i, j] = np.sum(-exog[:, i, None] * exog[:, j, None] * const_arr, axis=0).squeeze()
    tri_idx = np.triu_indices(dim, k=1)
    hess_arr[tri_idx] = hess_arr.T[tri_idx]
    da1 = -alpha ** (-2)
    dldpda = -np.sum(mu * exog * (y - mu) * a1 ** 2 / (mu + a1) ** 2, axis=0)
    hess_arr[-1, :-1] = dldpda
    hess_arr[:-1, -1] = dldpda
    da2 = 2 * alpha ** (-3)
    dalpha = da1 * (dgpart + np.log(prob) - (y - mu) / (a1 + mu))
    dada = (da2 * dalpha / da1 + da1 ** 2 * (special.polygamma(1, a1 + y) - special.polygamma(1, a1) + 1 / a1 - 1 / (a1 + mu) + (y - mu) / (mu + a1) ** 2)).sum()
    hess_arr[-1, -1] = dada
    return hess_arr