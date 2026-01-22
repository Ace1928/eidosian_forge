import numpy as np
from scipy.stats import norm as Gaussian
from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like
from . import norms
from ._qn import _qn
def _qn_naive(a, c=1 / (np.sqrt(2) * Gaussian.ppf(5 / 8))):
    """
    A naive implementation of the Qn robust estimator of scale, used solely
    to test the faster, more involved one

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        1/(np.sqrt(2) * scipy.stats.norm.ppf(5/8)), which is 2.219144.

    Returns
    -------
    The Qn robust estimator of scale
    """
    a = np.squeeze(a)
    n = a.shape[0]
    if a.size == 0:
        return np.nan
    else:
        h = int(n // 2 + 1)
        k = int(h * (h - 1) / 2)
        idx = np.triu_indices(n, k=1)
        diffs = np.abs(a[idx[0]] - a[idx[1]])
        output = np.partition(diffs, kth=k - 1)[k - 1]
        output = c * output
        return output