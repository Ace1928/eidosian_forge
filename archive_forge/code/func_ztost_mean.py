import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ztost_mean(self, low, upp):
    """test of (non-)equivalence of one sample, based on z-test

        TOST: two one-sided z-tests

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
        t1, pv1 : tuple
            test statistic and p-value for lower threshold test
        t2, pv2 : tuple
            test statistic and p-value for upper threshold test

        """
    t1, pv1 = self.ztest_mean(low, alternative='larger')
    t2, pv2 = self.ztest_mean(upp, alternative='smaller')
    return (np.maximum(pv1, pv2), (t1, pv1), (t2, pv2))