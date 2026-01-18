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
def tost_proportions_2indep(count1, nobs1, count2, nobs2, low, upp, method=None, compare='diff', correction=True):
    """
    Equivalence test based on two one-sided `test_proportions_2indep`

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis for equivalence testing are

    for compare = 'diff'

    - H0: prop1 - prop2 <= low or upp <= prop1 - prop2
    - H1: low < prop1 - prop2 < upp

    for compare = 'ratio'

    - H0: prop1 / prop2 <= low or upp <= prop1 / prop2
    - H1: low < prop1 / prop2 < upp


    for compare = 'odds-ratio'

    - H0: or <= low or upp <= or
    - H1: low < or < upp

    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))

    Parameters
    ----------
    count1, nobs1 :
        count and sample size for first sample
    count2, nobs2 :
        count and sample size for the second sample
    low, upp :
        equivalence margin for diff, risk ratio or odds ratio
    method : string
        method for computing the hypothesis test. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:
         - 'wald',
         - 'agresti-caffo'
         - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.

        ratio:
         - 'log': wald test using log transformation
         - 'log-adjusted': wald test using log transformation,
            adds 0.5 to counts
         - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.

        odds-ratio:
         - 'logit': wald test using logit transformation
         - 'logit-adjusted': : wald test using logit transformation,
            adds 0.5 to counts
         - 'logit-smoothed': : wald test using logit transformation, biases
            cell counts towards independence by adding two observations in
            total.
         - 'score' if correction is True, then this uses the degrees of freedom
            correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is `diff`, then the hypothesis test is for
        diff = p1 - p2.
        If compare is `ratio`, then the hypothesis test is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the hypothesis test is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.

    Returns
    -------
    pvalue : float
        p-value is the max of the pvalues of the two one-sided tests
    t1 : test results
        results instance for one-sided hypothesis at the lower margin
    t1 : test results
        results instance for one-sided hypothesis at the upper margin

    See Also
    --------
    test_proportions_2indep
    confint_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.

    The TOST equivalence test delegates to `test_proportions_2indep` and has
    the same method and comparison options.

    """
    tt1 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=low, method=method, compare=compare, alternative='larger', correction=correction, return_results=True)
    tt2 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=upp, method=method, compare=compare, alternative='smaller', correction=correction, return_results=True)
    idx_max = np.asarray(tt1.pvalue < tt2.pvalue, int)
    statistic = np.choose(idx_max, [tt1.statistic, tt2.statistic])
    pvalue = np.choose(idx_max, [tt1.pvalue, tt2.pvalue])
    res = HolderTuple(statistic=statistic, pvalue=pvalue, compare=compare, method=method, results_larger=tt1, results_smaller=tt2, title='Equivalence test for 2 independent proportions')
    return res