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
@Appender(_get_start_params_null_docs)
def _get_start_params_null(self):
    offset = getattr(self, 'offset', 0)
    exposure = getattr(self, 'exposure', 0)
    const = (self.endog / np.exp(offset + exposure)).mean()
    params = [np.log(const)]
    mu = const * np.exp(offset + exposure)
    resid = self.endog - mu
    a = self._estimate_dispersion(mu, resid, df_resid=resid.shape[0] - 1)
    params.append(a)
    return np.array(params)