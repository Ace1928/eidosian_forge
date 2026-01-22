import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def _HCCM(results, scale):
    """
    sandwich with pinv(x) * diag(scale) * pinv(x).T

    where pinv(x) = (X'X)^(-1) X
    and scale is (nobs,)
    """
    H = np.dot(results.model.pinv_wexog, scale[:, None] * results.model.pinv_wexog.T)
    return H