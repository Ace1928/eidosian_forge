import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def _test_solve_interior_point_2(self, linear_solver):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, 4))
    m.obj = pyo.Objective(expr=m.x ** 2)
    interface = InteriorPointInterface(m)
    ip_solver = InteriorPointSolver(linear_solver)
    status = ip_solver.solve(interface)
    self.assertEqual(status, InteriorPointStatus.optimal)
    interface.load_primals_into_pyomo_model()
    self.assertAlmostEqual(m.x.value, 1)