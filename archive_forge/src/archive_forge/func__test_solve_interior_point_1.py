import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def _test_solve_interior_point_1(self, linear_solver):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.obj = pyo.Objective(expr=m.x ** 2 + m.y ** 2)
    m.c1 = pyo.Constraint(expr=m.y == pyo.exp(m.x))
    m.c2 = pyo.Constraint(expr=m.y >= (m.x - 1) ** 2)
    interface = InteriorPointInterface(m)
    ip_solver = InteriorPointSolver(linear_solver)
    status = ip_solver.solve(interface)
    self.assertEqual(status, InteriorPointStatus.optimal)
    x = interface.get_primals()
    duals_eq = interface.get_duals_eq()
    duals_ineq = interface.get_duals_ineq()
    self.assertAlmostEqual(x[0], 0)
    self.assertAlmostEqual(x[1], 1)
    self.assertAlmostEqual(duals_eq[0], -1 - 1.0 / 3.0)
    self.assertAlmostEqual(duals_ineq[0], 2.0 / 3.0)
    interface.load_primals_into_pyomo_model()
    self.assertAlmostEqual(m.x.value, 0)
    self.assertAlmostEqual(m.y.value, 1)