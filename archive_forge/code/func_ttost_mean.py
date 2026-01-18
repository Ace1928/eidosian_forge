import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ttost_mean(self, low, upp):
    """test of (non-)equivalence of one sample

        TOST: two one-sided t tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1, df1 : tuple
            test statistic, pvalue and degrees of freedom for lower threshold
            test
        t2, pv2, df2 : tuple
            test statistic, pvalue and degrees of freedom for upper threshold
            test

        """
    t1, pv1, df1 = self.ttest_mean(low, alternative='larger')
    t2, pv2, df2 = self.ttest_mean(upp, alternative='smaller')
    return (np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2))