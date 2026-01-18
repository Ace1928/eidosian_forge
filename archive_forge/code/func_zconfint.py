import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def zconfint(x1, x2=None, value=0, alpha=0.05, alternative='two-sided', usevar='pooled', ddof=1.0):
    """confidence interval based on normal distribution z-test

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    value : float
        In the one sample case, value is the mean of x1 under the Null
        hypothesis.
        In the two sample case, value is the difference between mean of x1 and
        mean of x2 under the Null hypothesis. The test statistic is
        `x1_mean - x2_mean - value`.
    usevar : str, 'pooled'
        Currently, only 'pooled' is implemented.
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. see CompareMeans.ztest_ind for different options.
    ddof : int
        Degrees of freedom use in the calculation of the variance of the mean
        estimate. In the case of comparing means this is one, however it can
        be adjusted for testing other statistics (proportion, correlation)

    Notes
    -----
    checked only for 1 sample case

    usevar not implemented, is always pooled in two sample case

    ``value`` shifts the confidence interval so it is centered at
    `x1_mean - x2_mean - value`

    See Also
    --------
    ztest
    CompareMeans

    """
    if usevar != 'pooled':
        raise NotImplementedError('only usevar="pooled" is implemented')
    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0)
    if x2 is not None:
        x2 = np.asarray(x2)
        nobs2 = x2.shape[0]
        x2_mean = x2.mean(0)
        x2_var = x2.var(0)
        var_pooled = nobs1 * x1_var + nobs2 * x2_var
        var_pooled /= nobs1 + nobs2 - 2 * ddof
        var_pooled *= 1.0 / nobs1 + 1.0 / nobs2
    else:
        var_pooled = x1_var / (nobs1 - ddof)
        x2_mean = 0
    std_diff = np.sqrt(var_pooled)
    ci = _zconfint_generic(x1_mean - x2_mean - value, std_diff, alpha, alternative)
    return ci