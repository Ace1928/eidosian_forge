import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
def _cusum_squares_significance_bounds(self, alpha, points=None):
    """
        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lww, uww) because they use a different method for computing the
        critical value; in particular, they use tabled values from
        Table C, pp. 364-365 of "The Econometric Analysis of Time Series"
        Harvey, (1990), and use the value given to 99 observations for any
        larger number of observations. In contrast, we use the approximating
        critical values suggested in Edgerton and Wells (1994) which allows
        computing relatively good approximations for any number of
        observations.
        """
    d = max(self.nobs_diffuse, self.loglikelihood_burn)
    n = 0.5 * (self.nobs - d) - 1
    try:
        ix = [0.1, 0.05, 0.025, 0.01, 0.005].index(alpha / 2)
    except ValueError:
        raise ValueError('Invalid significance level.')
    scalars = _cusum_squares_scalars[:, ix]
    crit = scalars[0] / n ** 0.5 + scalars[1] / n + scalars[2] / n ** 1.5
    if points is None:
        points = np.array([d, self.nobs])
    line = (points - d) / (self.nobs - d)
    return (line - crit, line + crit)