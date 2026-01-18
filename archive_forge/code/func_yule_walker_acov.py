import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
def yule_walker_acov(acov, order=1, method='unbiased', df=None, inv=False):
    """
    Estimate AR(p) parameters from acovf using Yule-Walker equation.


    Parameters
    ----------
    acov : array_like, 1d
        auto-covariance
    order : int, optional
        The order of the autoregressive process.  Default is 1.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is False.

    Returns
    -------
    rho : ndarray
        The estimated autoregressive coefficients
    sigma
        TODO
    Rinv : ndarray
        inverse of the Toepliz matrix
    """
    return yule_walker(acov, order=order, method=method, df=df, inv=inv, demean=False)