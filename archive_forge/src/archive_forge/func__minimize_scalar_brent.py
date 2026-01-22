import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def _minimize_scalar_brent(func, brack=None, args=(), xtol=1.48e-08, maxiter=500, disp=0, **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    """
    _check_unknown_options(unknown_options)
    tol = xtol
    if tol < 0:
        raise ValueError('tolerance should be >= 0, got %r' % tol)
    brent = Brent(func=func, args=args, tol=tol, full_output=True, maxiter=maxiter, disp=disp)
    brent.set_bracket(brack)
    brent.optimize()
    x, fval, nit, nfev = brent.get_result(full_output=True)
    success = nit < maxiter and (not (np.isnan(x) or np.isnan(fval)))
    if success:
        message = f'\nOptimization terminated successfully;\nThe returned value satisfies the termination criteria\n(using xtol = {xtol} )'
    else:
        if nit >= maxiter:
            message = '\nMaximum number of iterations exceeded'
        if np.isnan(x) or np.isnan(fval):
            message = f'{_status_message['nan']}'
    if disp:
        _print_success_message_or_warn(not success, message)
    return OptimizeResult(fun=fval, x=x, nit=nit, nfev=nfev, success=success, message=message)