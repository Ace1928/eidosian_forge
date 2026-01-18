import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
summary table for probability that random draw x1 is larger than x2

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals. Coverage is 1 - alpha
        xname : None or list of str
            If None, then each row has a name column with generic names.
            If xname is a list of strings, then it will be included as part
            of those names.

        Returns
        -------
        SimpleTable instance with methods to convert to different output
        formats.
        