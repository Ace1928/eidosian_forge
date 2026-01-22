import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
def _root_scalar_newton_doc():
    """
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass