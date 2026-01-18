import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def test_prob_superior(self, value=0.5, alternative='two-sided'):
    """test for superiority probability

        H0: P(x1 > x2) + 0.5 * P(x1 = x2) = value

        The alternative is that the probability is either not equal, larger
        or smaller than the null-value depending on the chosen alternative.

        Parameters
        ----------
        value : float
            Value of the probability under the Null hypothesis.
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        res : HolderTuple
            HolderTuple instance with the following main attributes

            statistic : float
                Test statistic for z- or t-test
            pvalue : float
                Pvalue of the test based on either normal or t distribution.

        """
    p0 = value
    std_diff = np.sqrt(self.var / self.nobs)
    if not self.use_t:
        stat, pv = _zstat_generic(self.prob1, p0, std_diff, alternative, diff=0)
        distr = 'normal'
    else:
        stat, pv = _tstat_generic(self.prob1, p0, std_diff, self.df, alternative, diff=0)
        distr = 't'
    res = HolderTuple(statistic=stat, pvalue=pv, df=self.df, distribution=distr)
    return res