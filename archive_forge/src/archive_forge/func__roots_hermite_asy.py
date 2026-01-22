import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def _roots_hermite_asy(n):
    """Gauss-Hermite (physicist's) quadrature for large n.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    The sample points are the roots of the nth degree Hermite polynomial,
    :math:`H_n(x)`. These sample points and weights correctly integrate
    polynomials of degree :math:`2n - 1` or less over the interval
    :math:`[-\\infty, \\infty]` with weight function :math:`f(x) = e^{-x^2}`.

    This method relies on asymptotic expansions which work best for n > 150.
    The algorithm has linear runtime making computation for very large n
    feasible.

    Parameters
    ----------
    n : int
        quadrature order

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite

    References
    ----------
    .. [townsend.trogdon.olver-2014]
       Townsend, A. and Trogdon, T. and Olver, S. (2014)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*. :arXiv:`1410.5286`.

    .. [townsend.trogdon.olver-2015]
       Townsend, A. and Trogdon, T. and Olver, S. (2015)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*.
       IMA Journal of Numerical Analysis
       :doi:`10.1093/imanum/drv002`.
    """
    iv = _initial_nodes(n)
    nodes, weights = _newton(n, iv)
    if n % 2 == 0:
        nodes = hstack([-nodes[::-1], nodes])
        weights = hstack([weights[::-1], weights])
    else:
        nodes = hstack([-nodes[-1:0:-1], nodes])
        weights = hstack([weights[-1:0:-1], weights])
    weights *= sqrt(pi) / sum(weights)
    return (nodes, weights)