import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def _test_regularization_2(self, linear_solver):
    m = make_model_2()
    interface = InteriorPointInterface(m)
    ip_solver = InteriorPointSolver(linear_solver)
    status = ip_solver.solve(interface)
    self.assertEqual(status, InteriorPointStatus.optimal)
    interface.load_primals_into_pyomo_model()
    self.assertAlmostEqual(m.x.value, 1)
    self.assertAlmostEqual(m.y.value, pyo.exp(-1))