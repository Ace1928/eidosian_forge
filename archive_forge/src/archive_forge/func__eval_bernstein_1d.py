import warnings
import numpy as np
from scipy import interpolate, stats
def _eval_bernstein_1d(x, fvals, method='binom'):
    """Evaluate 1-dimensional bernstein polynomial given grid of values.

    experimental, comparing methods

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.
    method: "binom", "beta" or "bpoly"
        Method to construct Bernstein polynomial basis, used for comparison
        of parameterizations.

        - "binom" uses pmf of Binomial distribution
        - "beta" uses pdf of Beta distribution
        - "bpoly" uses one interval in scipy.interpolate.BPoly

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    k_terms = fvals.shape[-1]
    xx = np.asarray(x)
    k = np.arange(k_terms).astype(float)
    n = k_terms - 1.0
    if method.lower() == 'binom':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            poly_base = stats.binom.pmf(k, n, xx[..., None])
        bp_values = (fvals * poly_base).sum(-1)
    elif method.lower() == 'bpoly':
        bpb = interpolate.BPoly(fvals[:, None], [0.0, 1])
        bp_values = bpb(x)
    elif method.lower() == 'beta':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            poly_base = stats.beta.pdf(xx[..., None], k + 1, n - k + 1) / (n + 1)
        bp_values = (fvals * poly_base).sum(-1)
    else:
        raise ValueError('method not recogized')
    return bp_values