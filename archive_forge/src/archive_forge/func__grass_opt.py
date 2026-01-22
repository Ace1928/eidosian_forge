import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _grass_opt(params, fun, grad, maxiter, gtol):
    """
    Minimize a function on a Grassmann manifold.

    Parameters
    ----------
    params : array_like
        Starting value for the optimization.
    fun : function
        The function to be minimized.
    grad : function
        The gradient of fun.
    maxiter : int
        The maximum number of iterations.
    gtol : float
        Convergence occurs when the gradient norm falls below this value.

    Returns
    -------
    params : array_like
        The minimizing value for the objective function.
    fval : float
        The smallest achieved value of the objective function.
    cnvrg : bool
        True if the algorithm converged to a limit point.

    Notes
    -----
    `params` is 2-d, but `fun` and `grad` should take 1-d arrays
    `params.ravel()` as arguments.

    Reference
    ---------
    A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
    orthogonality constraints. SIAM J Matrix Anal Appl.
    http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
    """
    p, d = params.shape
    params = params.ravel()
    f0 = fun(params)
    cnvrg = False
    for _ in range(maxiter):
        g = grad(params)
        g -= np.dot(g, params) * params / np.dot(params, params)
        if np.sqrt(np.sum(g * g)) < gtol:
            cnvrg = True
            break
        gm = g.reshape((p, d))
        u, s, vt = np.linalg.svd(gm, 0)
        paramsm = params.reshape((p, d))
        pa0 = np.dot(paramsm, vt.T)

        def geo(t):
            pa = pa0 * np.cos(s * t) + u * np.sin(s * t)
            return np.dot(pa, vt).ravel()
        step = 2.0
        while step > 1e-10:
            pa = geo(-step)
            f1 = fun(pa)
            if f1 < f0:
                params = pa
                f0 = f1
                break
            step /= 2
    params = params.reshape((p, d))
    return (params, f0, cnvrg)