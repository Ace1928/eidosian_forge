import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def _initial_values(self, tr_radius):
    """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """
    lambda_ub = max(0, self.jac_mag / tr_radius + min(-self.hess_gershgorin_lb, self.hess_fro, self.hess_inf))
    lambda_lb = max(0, -min(self.hess.diagonal()), self.jac_mag / tr_radius - min(self.hess_gershgorin_ub, self.hess_fro, self.hess_inf))
    if tr_radius < self.previous_tr_radius:
        lambda_lb = max(self.lambda_lb, lambda_lb)
    if lambda_lb == 0:
        lambda_initial = 0
    else:
        lambda_initial = max(np.sqrt(lambda_lb * lambda_ub), lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))
    return (lambda_initial, lambda_lb, lambda_ub)