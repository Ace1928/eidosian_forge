import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_with_scalar_model_bad_starting_point(self):
    m, _ = make_scalar_model()
    solver = pyo.SolverFactory('scipy.fsolve')
    res = solver.solve(m)
    predicted_x = 4.90547401
    self.assertNotEqual(predicted_x, m.x.value)