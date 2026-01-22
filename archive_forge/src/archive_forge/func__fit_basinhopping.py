from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
def _fit_basinhopping(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Basin-hopping algorithm.

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

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    check_kwargs(kwargs, ('niter', 'niter_success', 'T', 'stepsize', 'interval', 'minimizer', 'seed'), 'basinhopping')
    kwargs = {k: v for k, v in kwargs.items()}
    niter = kwargs.setdefault('niter', 100)
    niter_success = kwargs.setdefault('niter_success', None)
    T = kwargs.setdefault('T', 1.0)
    stepsize = kwargs.setdefault('stepsize', 0.5)
    interval = kwargs.setdefault('interval', 50)
    seed = kwargs.get('seed')
    minimizer_kwargs = kwargs.get('minimizer', {})
    minimizer_kwargs['args'] = fargs
    minimizer_kwargs['jac'] = score
    method = minimizer_kwargs.get('method', None)
    if method and method != 'L-BFGS-B':
        minimizer_kwargs['hess'] = hess
    retvals = optimize.basinhopping(f, start_params, minimizer_kwargs=minimizer_kwargs, niter=niter, niter_success=niter_success, T=T, stepsize=stepsize, disp=disp, callback=callback, interval=interval, seed=seed)
    xopt = retvals.x
    if full_output:
        retvals = {'fopt': retvals.fun, 'iterations': retvals.nit, 'fcalls': retvals.nfev, 'converged': 'completed successfully' in retvals.message[0]}
    else:
        retvals = None
    return (xopt, retvals)