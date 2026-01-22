from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def _table_proportion(count, nobs):
    """
    Create a k by 2 contingency table for proportion

    helper function for proportions_chisquare

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.

    Returns
    -------
    table : ndarray
        (k, 2) contingency table

    Notes
    -----
    recent scipy has more elaborate contingency table functions

    """
    count = np.asarray(count)
    dt = np.promote_types(count.dtype, np.float64)
    count = np.asarray(count, dtype=dt)
    table = np.column_stack((count, nobs - count))
    expected = table.sum(0) * table.sum(1)[:, None] * 1.0 / table.sum()
    n_rows = table.shape[0]
    return (table, expected, n_rows)