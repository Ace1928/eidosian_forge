import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def _zstat_generic2(value, std, alternative):
    """generic (normal) z-test based on summary statistic

    The test statistic is :
        zstat = value / std

    and is assumed to be normally distributed with standard deviation ``std``.

    Parameters
    ----------
    value : float or ndarray
        Value of a sample statistic, for example mean.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std : float or ndarray
        Standard error of the sample statistic value.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    zstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        normally distributed.
    """
    zstat = value / std
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ['larger', 'l']:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ['smaller', 's']:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError('invalid alternative')
    return (zstat, pvalue)