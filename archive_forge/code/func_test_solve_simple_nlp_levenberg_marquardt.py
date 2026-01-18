import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_solve_simple_nlp_levenberg_marquardt(self):
    m, _ = make_simple_model()
    solver = pyo.SolverFactory('scipy.root')
    solver.set_options(dict(tol=1e-07, method='lm'))
    results = solver.solve(m)
    solution = [m.x[1].value, m.x[2].value, m.x[3].value]
    predicted = [0.92846891, -0.22610731, 0.29465397]
    self.assertStructuredAlmostEqual(solution, predicted)