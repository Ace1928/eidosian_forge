from warnings import warn
import numpy as np
from ._optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._slsqp_py import _minimize_slsqp
from ._constraints import (old_bound_to_new, new_bounds_to_old,
from ._differentiable_functions import FD_METHODS
def _optimize_result_for_equal_bounds(fun, bounds, method, args=(), constraints=()):
    """
    Provides a default OptimizeResult for when a bounded minimization method
    has (lb == ub).all().

    Parameters
    ----------
    fun: callable
    bounds: Bounds
    method: str
    constraints: Constraint
    """
    success = True
    message = 'All independent variables were fixed by bounds.'
    x0 = bounds.lb
    if constraints:
        message = 'All independent variables were fixed by bounds at values that satisfy the constraints.'
        constraints = standardize_constraints(constraints, x0, 'new')
    maxcv = 0
    for c in constraints:
        pc = PreparedConstraint(c, x0)
        violation = pc.violation(x0)
        if np.sum(violation):
            maxcv = max(maxcv, np.max(violation))
            success = False
            message = f'All independent variables were fixed by bounds, but the independent variables do not satisfy the constraints exactly. (Maximum violation: {maxcv}).'
    return OptimizeResult(x=x0, fun=fun(x0, *args), success=success, message=message, nfev=1, njev=0, nhev=0)