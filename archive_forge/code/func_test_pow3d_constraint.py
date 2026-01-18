import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_pow3d_constraint(self) -> None:
    n = 3
    np.random.seed(0)
    alpha = 0.275
    x, y, z = (Variable(n), Variable(n), Variable(n))
    con = PowCone3D(x, y, z, alpha)
    x0, y0 = (0.1 + np.random.rand(n), 0.1 + np.random.rand(n))
    z0 = x0 ** alpha * y0 ** (1 - alpha)
    z0[1] *= -1
    x.value, y.value, z.value = (x0, y0, z0)
    viol = con.residual
    self.assertLessEqual(viol, 1e-07)
    x1 = x0.copy()
    x1[0] *= -0.9
    x.value = x1
    viol = con.residual
    self.assertGreaterEqual(viol, 0.99 * abs(x1[0]))
    with self.assertRaises(ValueError):
        con = PowCone3D(x, y, z, 1.001)
    with self.assertRaises(ValueError):
        con = PowCone3D(x, y, z, -1e-05)