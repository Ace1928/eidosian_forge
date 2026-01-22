from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
def _fit_minimize(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using scipy minimize, where kwarg `min_method` defines the algorithm.

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
    kwargs.setdefault('min_method', 'BFGS')
    filter_opts = ['extra_fit_funcs', 'niter', 'min_method', 'tol', 'bounds', 'constraints']
    options = {k: v for k, v in kwargs.items() if k not in filter_opts}
    options['disp'] = disp
    options['maxiter'] = maxiter
    no_hess = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'COBYLA', 'SLSQP']
    no_jac = ['Nelder-Mead', 'Powell', 'COBYLA']
    if kwargs['min_method'] in no_hess:
        hess = None
    if kwargs['min_method'] in no_jac:
        score = None
    has_bounds = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
    if not SP_LT_15:
        has_bounds += ['Powell']
    if not SP_LT_17:
        has_bounds += ['Nelder-Mead']
    has_constraints = ['COBYLA', 'SLSQP', 'trust-constr']
    if 'bounds' in kwargs.keys() and kwargs['min_method'] in has_bounds:
        bounds = kwargs['bounds']
    else:
        bounds = None
    if 'constraints' in kwargs.keys() and kwargs['min_method'] in has_constraints:
        constraints = kwargs['constraints']
    else:
        constraints = ()
    res = optimize.minimize(f, start_params, args=fargs, method=kwargs['min_method'], jac=score, hess=hess, bounds=bounds, constraints=constraints, callback=callback, options=options)
    xopt = res.x
    retvals = None
    if full_output:
        nit = getattr(res, 'nit', np.nan)
        retvals = {'fopt': res.fun, 'iterations': nit, 'fcalls': res.nfev, 'warnflag': res.status, 'converged': res.success}
        if retall:
            retvals.update({'allvecs': res.values()})
    return (xopt, retvals)