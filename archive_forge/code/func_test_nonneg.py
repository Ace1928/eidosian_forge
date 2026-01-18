import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_nonneg(self) -> None:
    """Solve a trivial NonNeg-constrained problem through
        the conic and QP code paths.
        """
    x = cp.Variable(3)
    c = np.arange(3)
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.NonNeg(x - c)])
    prob.solve(solver=cp.ECOS)
    self.assertItemsAlmostEqual(x.value, c)
    prob.solve(solver=cp.OSQP)
    self.assertItemsAlmostEqual(x.value, c)