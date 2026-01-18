import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def tost_prob_superior(self, low, upp):
    """test of stochastic (non-)equivalence of p = P(x1 > x2)

        Null hypothesis:  p < low or p > upp
        Alternative hypothesis:  low < p < upp

        where p is the probability that a random draw from the population of
        the first sample has a larger value than a random draw from the
        population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the probability p that distribution 1 is stochastically
        superior to distribution 2 is outside of the interval given by
        thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        res : HolderTuple
            HolderTuple instance with the following main attributes

            pvalue : float
                Pvalue of the equivalence test given by the larger pvalue of
                the two one-sided tests.
            statistic : float
                Test statistic of the one-sided test that has the larger
                pvalue.
            results_larger : HolderTuple
                Results instanc with test statistic, pvalue and degrees of
                freedom for lower threshold test.
            results_smaller : HolderTuple
                Results instanc with test statistic, pvalue and degrees of
                freedom for upper threshold test.

        """
    t1 = self.test_prob_superior(low, alternative='larger')
    t2 = self.test_prob_superior(upp, alternative='smaller')
    idx_max = np.asarray(t1.pvalue < t2.pvalue, int)
    title = 'Equivalence test for Prob(x1 > x2) + 0.5 Prob(x1 = x2) '
    res = HolderTuple(statistic=np.choose(idx_max, [t1.statistic, t2.statistic]), pvalue=np.choose(idx_max, [t1.pvalue, t2.pvalue]), results_larger=t1, results_smaller=t2, title=title)
    return res