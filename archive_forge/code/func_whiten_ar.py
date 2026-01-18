import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
def whiten_ar(x, ar_coefs, order):
    """
    Whiten a series of columns according to an AR(p) covariance structure.

    This drops the initial conditions (Cochran-Orcut ?)
    Uses loop, so for short ar polynomials only, use lfilter otherwise

    This needs to improve, option on method, full additional to conditional

    Parameters
    ----------
    x : array_like, (nobs,) or (nobs, k_vars)
        The data to be whitened along axis 0
    ar_coefs : ndarray
        coefficients of AR lag- polynomial,   TODO: ar or ar_coefs?
    order : int

    Returns
    -------
    x_new : ndarray
        transformed array
    """
    rho = ar_coefs
    x = np.array(x, np.float64)
    _x = x.copy()
    if x.ndim == 2:
        rho = rho[:, None]
    for i in range(order):
        _x[i + 1:] = _x[i + 1:] - rho[i] * x[0:-(i + 1)]
    return _x[order:]