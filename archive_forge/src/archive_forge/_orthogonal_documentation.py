import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
Shifted Legendre polynomial.

    Defined as :math:`P^*_n(x) = P_n(2x - 1)` for :math:`P_n` the nth
    Legendre polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Shifted Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P^*_n` are orthogonal over :math:`[0, 1]`
    with weight function 1.

    