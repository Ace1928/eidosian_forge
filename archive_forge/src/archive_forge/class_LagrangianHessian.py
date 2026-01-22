import time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .._differentiable_functions import VectorFunction
from .._constraints import (
from .._hessian_update_strategy import BFGS
from .._optimize import OptimizeResult
from .._differentiable_functions import ScalarFunction
from .equality_constrained_sqp import equality_constrained_sqp
from .canonical_constraint import (CanonicalConstraint,
from .tr_interior_point import tr_interior_point
from .report import BasicReport, SQPReport, IPReport
class LagrangianHessian:
    """The Hessian of the Lagrangian as LinearOperator.

    The Lagrangian is computed as the objective function plus all the
    constraints multiplied with some numbers (Lagrange multipliers).
    """

    def __init__(self, n, objective_hess, constraints_hess):
        self.n = n
        self.objective_hess = objective_hess
        self.constraints_hess = constraints_hess

    def __call__(self, x, v_eq=np.empty(0), v_ineq=np.empty(0)):
        H_objective = self.objective_hess(x)
        H_constraints = self.constraints_hess(x, v_eq, v_ineq)

        def matvec(p):
            return H_objective.dot(p) + H_constraints.dot(p)
        return LinearOperator((self.n, self.n), matvec)