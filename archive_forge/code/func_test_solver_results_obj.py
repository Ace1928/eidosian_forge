import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_solver_results_obj(self):
    m, _ = make_simple_model()
    solver = pyo.SolverFactory('scipy.root')
    solver.set_options(dict(tol=1e-07))
    results = solver.solve(m)
    solution = [m.x[1].value, m.x[2].value, m.x[3].value]
    predicted = [0.92846891, -0.22610731, 0.29465397]
    self.assertStructuredAlmostEqual(solution, predicted)
    self.assertEqual(results.problem.number_of_constraints, 3)
    self.assertEqual(results.problem.number_of_variables, 3)
    self.assertEqual(results.solver.return_code, 1)
    self.assertEqual(results.solver.termination_condition, pyo.TerminationCondition.feasible)
    self.assertEqual(results.solver.message, 'The solution converged.')