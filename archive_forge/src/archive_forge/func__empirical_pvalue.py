from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def _empirical_pvalue(x, y, B, n, stats):
    """Calculate the empirical p-value based on permutations of `y`'s rows

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    B : int
        The number of iterations when evaluating the null distribution.
    n : Number of observations found in each of `x` and `y`.
    stats: namedtuple
        The result obtained from calling ``distance_statistics(x, y)``.

    Returns
    -------
    test_statistic : float
        The empirical test statistic.
    pval : float
        The empirical p-value.

    """
    B = int(B) if B else int(np.floor(200 + 5000 / n))
    empirical_dist = _get_test_statistic_distribution(x, y, B)
    pval = 1 - np.searchsorted(sorted(empirical_dist), stats.test_statistic) / len(empirical_dist)
    test_statistic = stats.test_statistic
    return (test_statistic, pval)