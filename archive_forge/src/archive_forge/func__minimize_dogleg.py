import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def _minimize_dogleg(fun, x0, args=(), jac=None, hess=None, **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the dog-leg trust-region algorithm.

    Options
    -------
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.

    """
    if jac is None:
        raise ValueError('Jacobian is required for dogleg minimization')
    if not callable(hess):
        raise ValueError('Hessian is required for dogleg minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, subproblem=DoglegSubproblem, **trust_region_options)