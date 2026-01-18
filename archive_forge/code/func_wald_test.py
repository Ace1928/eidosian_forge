import numpy as np
from scipy import stats
def wald_test(self, value):
    """Joint hypothesis tests that H0: f(params) = value.

        The alternative hypothesis is two-sided H1: f(params) != value.

        Warning: this might be replaced with more general version that returns
        ContrastResults.
        currently uses chisquare distribution, use_f option not yet implemented

        Parameters
        ----------
        value : float or ndarray
            value of f(params) under the Null Hypothesis

        Returns
        -------
        statistic : float
            Value of the test statistic.
        pvalue : float
            The p-value for the hypothesis test, based and chisquare
            distribution and implies a two-sided hypothesis test
        """
    m = self.predicted()
    v = self.cov()
    df_constraints = np.size(m)
    diff = m - value
    lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
    return (lmstat, stats.chi2.sf(lmstat, df_constraints))