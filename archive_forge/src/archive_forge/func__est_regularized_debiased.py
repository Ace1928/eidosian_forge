from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _est_regularized_debiased(mod, mnum, partitions, fit_kwds=None, score_kwds=None, hess_kwds=None):
    """estimates the regularized fitted parameters, is the default
    estimation_method for class DistributedModel.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    mnum : scalar
        Index of current partition.
    partitions : scalar
        Total number of partitions.
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    A tuple of parameters for regularized fit
        An array-like object of the fitted parameters, params
        An array-like object for the gradient
        A list of array like objects for nodewise_row
        A list of array like objects for nodewise_weight
    """
    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds
    if fit_kwds is None:
        raise ValueError('_est_regularized_debiased currently ' + 'requires that fit_kwds not be None.')
    else:
        alpha = fit_kwds['alpha']
    if 'L1_wt' in fit_kwds:
        L1_wt = fit_kwds['L1_wt']
    else:
        L1_wt = 1
    nobs, p = mod.exog.shape
    p_part = int(np.ceil(1.0 * p / partitions))
    params = mod.fit_regularized(**fit_kwds).params
    grad = _calc_grad(mod, params, alpha, L1_wt, score_kwds) / nobs
    wexog = _calc_wdesign_mat(mod, params, hess_kwds)
    nodewise_row_l = []
    nodewise_weight_l = []
    for idx in range(mnum * p_part, min((mnum + 1) * p_part, p)):
        nodewise_row = _calc_nodewise_row(wexog, idx, alpha)
        nodewise_row_l.append(nodewise_row)
        nodewise_weight = _calc_nodewise_weight(wexog, nodewise_row, idx, alpha)
        nodewise_weight_l.append(nodewise_weight)
    return (params, grad, nodewise_row_l, nodewise_weight_l)