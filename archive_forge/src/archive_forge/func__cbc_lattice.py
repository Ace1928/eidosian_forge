import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _cbc_lattice(n_dim, n_qmc_samples):
    """Compute a QMC lattice generator using a Fast CBC construction.

    Parameters
    ----------
    n_dim : int > 0
        The number of dimensions for the lattice.
    n_qmc_samples : int > 0
        The desired number of QMC samples. This will be rounded down to the
        nearest prime to enable the CBC construction.

    Returns
    -------
    q : float array : shape=(n_dim,)
        The lattice generator vector. All values are in the open interval
        `(0, 1)`.
    actual_n_qmc_samples : int
        The prime number of QMC samples that must be used with this lattice,
        no more, no less.

    References
    ----------
    .. [1] Nuyens, D. and Cools, R. "Fast Component-by-Component Construction,
           a Reprise for Different Kernels", In H. Niederreiter and D. Talay,
           editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004,
           Springer-Verlag, 2006, 371-385.
    """
    primes = primes_from_2_to(n_qmc_samples + 1)
    n_qmc_samples = primes[-1]
    bt = np.ones(n_dim)
    gm = np.hstack([1.0, 0.8 ** np.arange(n_dim - 1)])
    q = 1
    w = 0
    z = np.arange(1, n_dim + 1)
    m = (n_qmc_samples - 1) // 2
    g = _primitive_root(n_qmc_samples)
    perm = np.ones(m, dtype=int)
    for j in range(m - 1):
        perm[j + 1] = g * perm[j] % n_qmc_samples
    perm = np.minimum(n_qmc_samples - perm, perm)
    pn = perm / n_qmc_samples
    c = pn * pn - pn + 1.0 / 6
    fc = fft(c)
    for s in range(1, n_dim):
        reordered = np.hstack([c[:w + 1][::-1], c[w + 1:m][::-1]])
        q = q * (bt[s - 1] + gm[s - 1] * reordered)
        w = ifft(fc * fft(q)).real.argmin()
        z[s] = perm[w]
    q = z / n_qmc_samples
    return (q, n_qmc_samples)