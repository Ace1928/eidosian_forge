import numpy as np
from scipy import special
from scipy.special import gammaln
def tstd_dlldy(y, df):
    """derivative of log pdf of standardized t with respect to y

        Parameters
        ----------
    y : array_like
        data points of random variable at which loglike is evaluated
    df : array_like
        degrees of freedom,shape parameters of log-likelihood function
        of t distribution

    Returns
    -------
    dlldy : ndarray
        derivative of loglikelihood wrt random variable y evaluated at the
        points given in y


    Notes
    -----
    parametrized for garch, standardized to variance=1
    """
    return -(df + 1) / (df - 2.0) / (1 + y ** 2 / (df - 2.0)) * y