from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta, gamma=0.0001, tau_min=0.1, tau_max=0.5):
    """
    Nonmonotone backtracking line search as described in [1]_

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    prev_fs : float
        List of previous merit function values. Should have ``len(prev_fs) <= M``
        where ``M`` is the nonmonotonicity window parameter.
    eta : float
        Allowed merit function increase, see [1]_
    gamma, tau_min, tau_max : float, optional
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

    References
    ----------
    [1] "Spectral residual method without gradient information for solving
        large-scale nonlinear systems of equations." W. La Cruz,
        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).

    """
    f_k = prev_fs[-1]
    f_bar = max(prev_fs)
    alpha_p = 1
    alpha_m = 1
    alpha = 1
    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)
        if fp <= f_bar + eta - gamma * alpha_p ** 2 * f_k:
            alpha = alpha_p
            break
        alpha_tp = alpha_p ** 2 * f_k / (fp + (2 * alpha_p - 1) * f_k)
        xp = x_k - alpha_m * d
        fp, Fp = f(xp)
        if fp <= f_bar + eta - gamma * alpha_m ** 2 * f_k:
            alpha = -alpha_m
            break
        alpha_tm = alpha_m ** 2 * f_k / (fp + (2 * alpha_m - 1) * f_k)
        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)
    return (alpha, xp, fp, Fp)