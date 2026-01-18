import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def weights_uniform(nlags):
    """uniform weights for HAC

    this will be moved to another module

    Parameters
    ----------
    nlags : int
       highest lag in the kernel window, this does not include the zero lag

    Returns
    -------
    kernel : ndarray, (nlags+1,)
        weights for uniform kernel

    """
    return np.ones(nlags + 1)