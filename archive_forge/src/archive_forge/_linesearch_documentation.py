from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np

    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    f_k : float
        Initial merit function value.
    C, Q : float
        Control parameters. On the first iteration, give values
        Q=1.0, C=f_k
    eta : float
        Allowed merit function increase, see [1]_
    nu, gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position
    C : float
        New value for the control parameter C
    Q : float
        New value for the control parameter Q

    References
    ----------
    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
           search and its application to the spectral residual
           method'', IMA J. Numer. Anal. 29, 814 (2009).

    