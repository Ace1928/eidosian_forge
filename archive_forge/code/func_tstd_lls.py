import numpy as np
from scipy import special
from scipy.special import gammaln
def tstd_lls(y, params, df):
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
    """
    mu, sigma2 = params.T
    df = df * 1.0
    lls = gammaln((df + 1) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log((df - 2) * np.pi)
    lls -= (df + 1) / 2.0 * np.log(1.0 + (y - mu) ** 2 / (df - 2) / sigma2) + 0.5 * np.log(sigma2)
    return lls