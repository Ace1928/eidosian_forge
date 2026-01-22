import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def _initial_nodes_b(n, k):
    """Gatteschi initial guesses

    Computes an initial approximation to the square of the kth
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.2 in the
    original paper. The guesses are accurate in the region just
    below :math:`\\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate root

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    nu = 4.0 * floor(n / 2.0) + 2.0 * a + 2.0
    ak = _specfun.airyzo(k.max(), 1)[0][::-1]
    xksq = nu + 2.0 ** (2.0 / 3.0) * ak * nu ** (1.0 / 3.0) + 1.0 / 5.0 * 2.0 ** (4.0 / 3.0) * ak ** 2 * nu ** (-1.0 / 3.0) + (9.0 / 140.0 - 12.0 / 175.0 * ak ** 3) * nu ** (-1.0) + (16.0 / 1575.0 * ak + 92.0 / 7875.0 * ak ** 4) * 2.0 ** (2.0 / 3.0) * nu ** (-5.0 / 3.0) - (15152.0 / 3031875.0 * ak ** 5 + 1088.0 / 121275.0 * ak ** 2) * 2.0 ** (1.0 / 3.0) * nu ** (-7.0 / 3.0)
    return xksq