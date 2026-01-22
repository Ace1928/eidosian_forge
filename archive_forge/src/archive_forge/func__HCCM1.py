import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def _HCCM1(results, scale):
    """
    sandwich with pinv(x) * scale * pinv(x).T

    where pinv(x) = (X'X)^(-1) X
    and scale is (nobs, nobs), or (nobs,) with diagonal matrix diag(scale)

    Parameters
    ----------
    results : result instance
       need to contain regression results, uses results.model.pinv_wexog
    scale : ndarray (nobs,) or (nobs, nobs)
       scale matrix, treated as diagonal matrix if scale is one-dimensional

    Returns
    -------
    H : ndarray (k_vars, k_vars)
        robust covariance matrix for the parameter estimates

    """
    if scale.ndim == 1:
        H = np.dot(results.model.pinv_wexog, scale[:, None] * results.model.pinv_wexog.T)
    else:
        H = np.dot(results.model.pinv_wexog, np.dot(scale, results.model.pinv_wexog.T))
    return H