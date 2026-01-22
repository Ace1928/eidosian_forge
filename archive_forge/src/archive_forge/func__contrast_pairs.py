import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def _contrast_pairs(k_params, k_level, idx_start):
    """create pairwise contrast for reference coding

    currently not used,
    using encoding contrast matrix is more general, but requires requires
    factor information from patsy design_info.


    Parameters
    ----------
    k_params : int
        number of parameters
    k_level : int
        number of levels or categories (including reference case)
    idx_start : int
        Index of the first parameter of this factor. The restrictions on the
        factor are inserted as a block in the full restriction matrix starting
        at column with index `idx_start`.

    Returns
    -------
    contrasts : ndarray
        restriction matrix with k_params columns and number of rows equal to
        the number of restrictions.
    """
    k_level_m1 = k_level - 1
    idx_pairs = np.triu_indices(k_level_m1, 1)
    k = len(idx_pairs[0])
    c_pairs = np.zeros((k, k_level_m1))
    c_pairs[np.arange(k), idx_pairs[0]] = -1
    c_pairs[np.arange(k), idx_pairs[1]] = 1
    c_reference = np.eye(k_level_m1)
    c = np.concatenate((c_reference, c_pairs), axis=0)
    k_all = c.shape[0]
    contrasts = np.zeros((k_all, k_params))
    contrasts[:, idx_start:idx_start + k_level_m1] = c
    return contrasts