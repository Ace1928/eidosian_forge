import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _mvn_qmc_integrand(covar, low, high, use_tent=False):
    """Transform the multivariate normal integration into a QMC integrand over
    a unit hypercube.

    The dimensionality of the resulting hypercube integration domain is one
    less than the dimensionality of the original integrand. Note that this
    transformation subsumes the integration bounds in order to account for
    infinite bounds. The QMC integration one does with the returned integrand
    should be on the unit hypercube.

    Parameters
    ----------
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    use_tent : bool, optional
        If True, then use tent periodization. Only helpful for lattice rules.

    Returns
    -------
    integrand : Callable[[NDArray], NDArray]
        The QMC-integrable integrand. It takes an
        ``(n_qmc_samples, ndim_integrand)`` array of QMC samples in the unit
        hypercube and returns the ``(n_qmc_samples,)`` evaluations of at these
        QMC points.
    ndim_integrand : int
        The dimensionality of the integrand. Equal to ``n-1``.
    """
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ndim_integrand = n - 1
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci

    def integrand(*zs):
        ndim_qmc = len(zs)
        n_qmc_samples = len(np.atleast_1d(zs[0]))
        assert ndim_qmc == ndim_integrand
        y = np.zeros((ndim_qmc, n_qmc_samples))
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            if use_tent:
                x = abs(2 * zs[i - 1] - 1)
            else:
                x = zs[i - 1]
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        return pv
    return (integrand, ndim_integrand)