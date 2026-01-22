import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def _gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds):
    """
    Negative penalized log-likelihood functions.

    Returns the negative penalized log-likelihood, its derivative, and
    its Hessian.  The penalty only includes the smooth (L2) term.

    All three functions have argument signature (x, model), where
    ``x`` is a point in the parameter space and ``model`` is an
    arbitrary statsmodels regression model.
    """

    def nploglike(params, model):
        nobs = model.nobs
        pen_llf = alpha[k] * (1 - L1_wt) * np.sum(params ** 2) / 2
        llf = model.loglike(np.r_[params], **loglike_kwds)
        return -llf / nobs + pen_llf

    def npscore(params, model):
        nobs = model.nobs
        pen_grad = alpha[k] * (1 - L1_wt) * params
        gr = -model.score(np.r_[params], **score_kwds)[0] / nobs
        return gr + pen_grad

    def nphess(params, model):
        nobs = model.nobs
        pen_hess = alpha[k] * (1 - L1_wt)
        h = -model.hessian(np.r_[params], **hess_kwds)[0, 0] / nobs + pen_hess
        return h
    return (nploglike, npscore, nphess)