import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
def approx_fprime_cs(x, f, epsilon=None, args=(), kwargs={}):
    """
    Calculate gradient or Jacobian with complex step derivative approximation

    Parameters
    ----------
    x : ndarray
        parameters at which the derivative is evaluated
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. Optimal step-size is
        EPS*x. See note.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.

    Returns
    -------
    partials : ndarray
       array of partial derivatives, Gradient or Jacobian

    Notes
    -----
    The complex-step derivative has truncation error O(epsilon**2), so
    truncation error can be eliminated by choosing epsilon to be very small.
    The complex-step derivative avoids the problem of round-off error with
    small epsilon because there is no subtraction.
    """
    n = len(x)
    epsilon = _get_epsilon(x, 1, epsilon, n)
    increments = np.identity(n) * 1j * epsilon
    partials = [f(x + ih, *args, **kwargs).imag / epsilon[i] for i, ih in enumerate(increments)]
    return np.array(partials).T