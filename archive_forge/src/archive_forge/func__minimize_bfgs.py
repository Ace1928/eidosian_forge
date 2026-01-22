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
def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None, gtol=1e-05, norm=np.inf, eps=_epsilon, maxiter=None, disp=False, return_all=False, finite_diff_rel_step=None, xrtol=0, c1=0.0001, c2=0.9, hess_inv0=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.
    hess_inv0 : None or ndarray, optional
        Initial inverse hessian estimate, shape (n, n). If None (default) then
        the identity matrix is used.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    If minimization doesn't complete successfully, with an error message of
    ``Desired error not necessarily achieved due to precision loss``, then
    consider setting `gtol` to a higher value. This precision loss typically
    occurs when the (finite difference) numerical differentiation cannot provide
    sufficient precision to satisfy the `gtol` termination criterion.
    This can happen when working in single precision and a callable jac is not
    provided. For single precision problems a `gtol` of 1e-3 seems to work.
    """
    _check_unknown_options(unknown_options)
    _check_positive_definite(hess_inv0)
    retall = return_all
    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200
    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps, finite_diff_rel_step=finite_diff_rel_step)
    f = sf.fun
    myfprime = sf.grad
    old_fval = f(x0)
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I if hess_inv0 is None else hess_inv0
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while gnorm > gtol and k < maxiter:
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e+100, c1=c1, c2=c2)
        except _LineSearchError:
            warnflag = 2
            break
        sk = alpha_k * pk
        xkp1 = xk + sk
        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        gnorm = vecnorm(gfk, ord=norm)
        if gnorm <= gtol:
            break
        if alpha_k * vecnorm(pk) <= xrtol * (xrtol + vecnorm(xk)):
            break
        if not np.isfinite(old_fval):
            warnflag = 2
            break
        rhok_inv = np.dot(yk, sk)
        if rhok_inv == 0.0:
            rhok = 1000.0
            if disp:
                msg = 'Divide-by-zero encountered: rhok assumed large'
                _print_success_message_or_warn(True, msg)
        else:
            rhok = 1.0 / rhok_inv
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + rhok * sk[:, np.newaxis] * sk[np.newaxis, :]
    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']
    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print('         Current function value: %f' % fval)
        print('         Iterations: %d' % k)
        print('         Function evaluations: %d' % sf.nfev)
        print('         Gradient evaluations: %d' % sf.ngev)
    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev, njev=sf.ngev, status=warnflag, success=warnflag == 0, message=msg, x=xk, nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result