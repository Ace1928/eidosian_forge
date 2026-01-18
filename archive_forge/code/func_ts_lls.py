import numpy as np
from scipy import special
from scipy.special import gammaln
def ts_lls(y, params, df):
    """t loglikelihood given observations and mean mu and variance sigma2 = 1

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows
    df : int
        degrees of freedom of the t distribution

    Returns
    -------
    lls : ndarray
        contribution to loglikelihood for each observation

    Notes
    -----
    parametrized for garch
    normalized/rescaled so that sigma2 is the variance

    >>> df = 10; sigma = 1.
    >>> stats.t.stats(df, loc=0., scale=sigma.*np.sqrt((df-2.)/df))
    (array(0.0), array(1.0))
    >>> sigma = np.sqrt(2.)
    >>> stats.t.stats(df, loc=0., scale=sigma*np.sqrt((df-2.)/df))
    (array(0.0), array(2.0))
    """
    print(y, params, df)
    mu, sigma2 = params.T
    df = df * 1.0
    lls = gammaln((df + 1) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi)
    lls -= (df + 1.0) / 2.0 * np.log(1.0 + (y - mu) ** 2 / df / sigma2) + 0.5 * np.log(sigma2)
    return lls