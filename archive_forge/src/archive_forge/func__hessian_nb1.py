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
def _hessian_nb1(self, params):
    """
        Hessian of NB1 model.
        """
    if self._transparams:
        alpha = np.exp(params[-1])
    else:
        alpha = params[-1]
    params = params[:-1]
    exog = self.exog
    y = self.endog[:, None]
    mu = self.predict(params)[:, None]
    a1 = mu / alpha
    dgpart = digamma(y + a1) - digamma(a1)
    prob = 1 / (1 + alpha)
    dim = exog.shape[1]
    hess_arr = np.empty((dim + 1, dim + 1))
    dparams = exog / alpha * (np.log(prob) + dgpart)
    dmudb = exog * mu
    xmu_alpha = exog * a1
    trigamma = special.polygamma(1, a1 + y) - special.polygamma(1, a1)
    for i in range(dim):
        for j in range(dim):
            if j > i:
                continue
            hess_arr[i, j] = np.squeeze(np.sum(dparams[:, i, None] * dmudb[:, j, None] + xmu_alpha[:, i, None] * xmu_alpha[:, j, None] * trigamma, axis=0))
    tri_idx = np.triu_indices(dim, k=1)
    hess_arr[tri_idx] = hess_arr.T[tri_idx]
    dldpda = np.sum(-a1 * dparams + exog * a1 * (-trigamma * mu / alpha ** 2 - prob), axis=0)
    hess_arr[-1, :-1] = dldpda
    hess_arr[:-1, -1] = dldpda
    log_alpha = np.log(prob)
    alpha3 = alpha ** 3
    alpha2 = alpha ** 2
    mu2 = mu ** 2
    dada = (alpha3 * mu * (2 * log_alpha + 2 * dgpart + 3) - 2 * alpha3 * y + 4 * alpha2 * mu * (log_alpha + dgpart) + alpha2 * (2 * mu - y) + 2 * alpha * mu2 * trigamma + mu2 * trigamma + alpha2 * mu2 * trigamma + 2 * alpha * mu * (log_alpha + dgpart)) / (alpha ** 4 * (alpha2 + 2 * alpha + 1))
    hess_arr[-1, -1] = dada.sum()
    return hess_arr