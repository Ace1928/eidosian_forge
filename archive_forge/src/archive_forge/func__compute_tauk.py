import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def _compute_tauk(n, k, maxit=5):
    """Helper function for Tricomi initial guesses

    For details, see formula 3.1 in lemma 3.1 in the
    original paper.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots :math:`	au_k` to compute
    maxit : int
        Number of Newton maxit performed, the default
        value of 5 is sufficient.

    Returns
    -------
    tauk : ndarray
        Roots of equation 3.1

    See Also
    --------
    initial_nodes_a
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    c = (4.0 * floor(n / 2.0) - 4.0 * k + 3.0) * pi / (4.0 * floor(n / 2.0) + 2.0 * a + 2.0)

    def f(x):
        return x - sin(x) - c

    def df(x):
        return 1.0 - cos(x)
    xi = 0.5 * pi
    for i in range(maxit):
        xi = xi - f(xi) / df(xi)
    return xi