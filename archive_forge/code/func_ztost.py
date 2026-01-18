import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ztost(x1, low, upp, x2=None, usevar='pooled', ddof=1.0):
    """Equivalence test based on normal distribution

    Parameters
    ----------
    x1 : array_like
        one sample or first sample for 2 independent samples
    low, upp : float
        equivalence interval low < m1 - m2 < upp
    x1 : array_like or None
        second sample for 2 independent samples test. If None, then a
        one-sample test is performed.
    usevar : str, 'pooled'
        If `pooled`, then the standard deviation of the samples is assumed to be
        the same. Only `pooled` is currently implemented.

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    checked only for 1 sample case

    """
    tt1 = ztest(x1, x2, alternative='larger', usevar=usevar, value=low, ddof=ddof)
    tt2 = ztest(x1, x2, alternative='smaller', usevar=usevar, value=upp, ddof=ddof)
    return (np.maximum(tt1[1], tt2[1]), tt1, tt2)