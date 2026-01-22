import warnings
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special
def _faa_di_bruno_partitions(n):
    """
    Return all non-negative integer solutions of the diophantine equation

            n*k_n + ... + 2*k_2 + 1*k_1 = n   (1)

    Parameters
    ----------
    n : int
        the r.h.s. of Eq. (1)

    Returns
    -------
    partitions : list
        Each solution is itself a list of the form `[(m, k_m), ...]`
        for non-zero `k_m`. Notice that the index `m` is 1-based.

    Examples:
    ---------
    >>> _faa_di_bruno_partitions(2)
    [[(1, 2)], [(2, 1)]]
    >>> for p in _faa_di_bruno_partitions(4):
    ...     assert 4 == sum(m * k for (m, k) in p)
    """
    if n < 1:
        raise ValueError('Expected a positive integer; got %s instead' % n)
    try:
        return _faa_di_bruno_cache[n]
    except KeyError:
        raise NotImplementedError('Higher order terms not yet implemented.')