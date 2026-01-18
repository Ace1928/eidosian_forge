import numpy as np
from cvxpy import Maximize, Minimize, Problem
from cvxpy.atoms import diag, exp, hstack, pnorm
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import SolverTestHelper
def test_nonneg_constraints_backend(self) -> None:
    x = Variable(shape=(2,), name='x')
    objective = Maximize(-4 * x[0] - 5 * x[1])
    constr_expr = hstack([3 - (2 * x[0] + x[1]), 3 - (x[0] + 2 * x[1]), x[0], x[1]])
    constraints = [NonNeg(constr_expr)]
    prob = Problem(objective, constraints)
    self.assertFalse(ConeMatrixStuffing().accepts(prob))
    self.assertTrue(FlipObjective().accepts(prob))
    p_min = FlipObjective().apply(prob)
    self.assertTrue(ConeMatrixStuffing().accepts(p_min[0]))