import math
import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def _minimize_trust_ncg(fun, x0, args=(), jac=None, hess=None, hessp=None, **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the Newton conjugate gradient trust-region algorithm.

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
        raise ValueError('Jacobian is required for Newton-CG trust-region minimization')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product is required for Newton-CG trust-region minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp, subproblem=CGSteihaugSubproblem, **trust_region_options)