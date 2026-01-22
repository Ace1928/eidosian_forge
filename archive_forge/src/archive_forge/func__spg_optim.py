import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def _spg_optim(func, grad, start, project, maxiter=10000.0, M=10, ctol=0.001, maxiter_nmls=200, lam_min=1e-30, lam_max=1e+30, sig1=0.1, sig2=0.9, gam=0.0001):
    """
    Implements the spectral projected gradient method for minimizing a
    differentiable function on a convex domain.

    Parameters
    ----------
    func : real valued function
        The objective function to be minimized.
    grad : real array-valued function
        The gradient of the objective function
    start : array_like
        The starting point
    project : function
        In-place projection of the argument to the domain
        of func.
    ... See notes regarding additional arguments

    Returns
    -------
    rslt : Bunch
        rslt.params is the final iterate, other fields describe
        convergence status.

    Notes
    -----
    This can be an effective heuristic algorithm for problems where no
    guaranteed algorithm for computing a global minimizer is known.

    There are a number of tuning parameters, but these generally
    should not be changed except for `maxiter` (positive integer) and
    `ctol` (small positive real).  See the Birgin et al reference for
    more information about the tuning parameters.

    Reference
    ---------
    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected
    gradient methods: Review and perspectives. Journal of Statistical
    Software (preprint).  Available at:
    http://www.ime.usp.br/~egbirgin/publications/bmr5.pdf
    """
    lam = min(10 * lam_min, lam_max)
    params = start.copy()
    gval = grad(params)
    obj_hist = [func(params)]
    for itr in range(int(maxiter)):
        df = params - gval
        project(df)
        df -= params
        if np.max(np.abs(df)) < ctol:
            return Bunch(**{'Converged': True, 'params': params, 'objective_values': obj_hist, 'Message': 'Converged successfully'})
        d = params - lam * gval
        project(d)
        d -= params
        alpha, params1, fval, gval1 = _nmono_linesearch(func, grad, params, d, obj_hist, M=M, sig1=sig1, sig2=sig2, gam=gam, maxiter=maxiter_nmls)
        if alpha is None:
            return Bunch(**{'Converged': False, 'params': params, 'objective_values': obj_hist, 'Message': 'Failed in nmono_linesearch'})
        obj_hist.append(fval)
        s = params1 - params
        y = gval1 - gval
        sy = (s * y).sum()
        if sy <= 0:
            lam = lam_max
        else:
            ss = (s * s).sum()
            lam = max(lam_min, min(ss / sy, lam_max))
        params = params1
        gval = gval1
    return Bunch(**{'Converged': False, 'params': params, 'objective_values': obj_hist, 'Message': 'spg_optim did not converge'})