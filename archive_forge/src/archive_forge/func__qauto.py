import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _qauto(func, covar, low, high, rng, error=0.001, limit=10000, **kwds):
    """Automatically rerun the integration to get the required error bound.

    Parameters
    ----------
    func : callable
        Either :func:`_qmvn` or :func:`_qmvt`.
    covar, low, high : array
        As specified in :func:`_qmvn` and :func:`_qmvt`.
    rng : Generator, optional
        default_rng(), yada, yada
    error : float > 0
        The desired error bound.
    limit : int > 0:
        The rough limit of the number of integration points to consider. The
        integration will stop looping once this limit has been *exceeded*.
    **kwds :
        Other keyword arguments to pass to `func`. When using :func:`_qmvt`, be
        sure to include ``nu=`` as one of these.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    n_samples : int
        The number of integration points actually used.
    """
    n = len(covar)
    n_samples = 0
    if n == 1:
        prob = phi(high) - phi(low)
        est_error = 1e-15
    else:
        mi = min(limit, n * 1000)
        prob = 0.0
        est_error = 1.0
        ei = 0.0
        while est_error > error and n_samples < limit:
            mi = round(np.sqrt(2) * mi)
            pi, ei, ni = func(mi, covar, low, high, rng=rng, **kwds)
            n_samples += ni
            wt = 1.0 / (1 + (ei / est_error) ** 2)
            prob += wt * (pi - prob)
            est_error = np.sqrt(wt) * ei
    return (prob, est_error, n_samples)