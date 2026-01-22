import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def _HCCM2(hessian_inv, scale):
    """
    sandwich with (X'X)^(-1) * scale * (X'X)^(-1)

    scale is (kvars, kvars)
    this uses results.normalized_cov_params for (X'X)^(-1)

    Parameters
    ----------
    results : result instance
       need to contain regression results, uses results.normalized_cov_params
    scale : ndarray (k_vars, k_vars)
       scale matrix

    Returns
    -------
    H : ndarray (k_vars, k_vars)
        robust covariance matrix for the parameter estimates

    """
    if scale.ndim == 1:
        scale = scale[:, None]
    xxi = hessian_inv
    H = np.dot(np.dot(xxi, scale), xxi.T)
    return H