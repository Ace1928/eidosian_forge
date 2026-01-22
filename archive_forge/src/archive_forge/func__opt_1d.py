import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def _opt_1d(func, grad, hess, model, start, L1_wt, tol, check_step=True):
    """
    One-dimensional helper for elastic net.

    Parameters
    ----------
    func : function
        A smooth function of a single variable to be optimized
        with L1 penaty.
    grad : function
        The gradient of `func`.
    hess : function
        The Hessian of `func`.
    model : statsmodels model
        The model being fit.
    start : real
        A starting value for the function argument
    L1_wt : non-negative real
        The weight for the L1 penalty function.
    tol : non-negative real
        A convergence threshold.
    check_step : bool
        If True, check that the first step is an improvement and
        use bisection if it is not.  If False, return after the
        first step regardless.

    Notes
    -----
    ``func``, ``grad``, and ``hess`` have argument signature (x,
    model), where ``x`` is a point in the parameter space and
    ``model`` is the model being fit.

    If the log-likelihood for the model is exactly quadratic, the
    global minimum is returned in one step.  Otherwise numerical
    bisection is used.

    Returns
    -------
    The argmin of the objective function.
    """
    x = start
    f = func(x, model)
    b = grad(x, model)
    c = hess(x, model)
    d = b - c * x
    if L1_wt > np.abs(d):
        return 0.0
    if d >= 0:
        h = (L1_wt - b) / c
    elif d < 0:
        h = -(L1_wt + b) / c
    else:
        return np.nan
    if not check_step:
        return x + h
    f1 = func(x + h, model) + L1_wt * np.abs(x + h)
    if f1 <= f + L1_wt * np.abs(x) + 1e-10:
        return x + h
    from scipy.optimize import brent
    x_opt = brent(func, args=(model,), brack=(x - 1, x + 1), tol=tol)
    return x_opt