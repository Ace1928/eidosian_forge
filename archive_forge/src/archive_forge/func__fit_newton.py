from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
def _fit_newton(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None, ridge_factor=1e-10):
    """
    Fit using Newton-Raphson algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.
    ridge_factor : float
        Regularization factor for Hessian matrix.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    check_kwargs(kwargs, ('tol', 'ridge_factor'), 'newton')
    tol = kwargs.setdefault('tol', 1e-08)
    ridge_factor = kwargs.setdefault('ridge_factor', 1e-10)
    iterations = 0
    oldparams = np.inf
    newparams = np.asarray(start_params)
    if retall:
        history = [oldparams, newparams]
    while iterations < maxiter and np.any(np.abs(newparams - oldparams) > tol):
        H = np.asarray(hess(newparams))
        if not np.all(ridge_factor == 0):
            H[np.diag_indices(H.shape[0])] += ridge_factor
        oldparams = newparams
        newparams = oldparams - np.linalg.solve(H, score(oldparams))
        if retall:
            history.append(newparams)
        if callback is not None:
            callback(newparams)
        iterations += 1
    fval = f(newparams, *fargs)
    if iterations == maxiter:
        warnflag = 1
        if disp:
            print('Warning: Maximum number of iterations has been exceeded.')
            print('         Current function value: %f' % fval)
            print('         Iterations: %d' % iterations)
    else:
        warnflag = 0
        if disp:
            print('Optimization terminated successfully.')
            print('         Current function value: %f' % fval)
            print('         Iterations %d' % iterations)
    if full_output:
        xopt, fopt, niter, gopt, hopt = (newparams, f(newparams, *fargs), iterations, score(newparams), hess(newparams))
        converged = not warnflag
        retvals = {'fopt': fopt, 'iterations': niter, 'score': gopt, 'Hessian': hopt, 'warnflag': warnflag, 'converged': converged}
        if retall:
            retvals.update({'allvecs': history})
    else:
        xopt = newparams
        retvals = None
    return (xopt, retvals)