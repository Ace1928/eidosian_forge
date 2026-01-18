import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_results_object(self):
    m, _ = make_scalar_model()
    solver = pyo.SolverFactory('scipy.secant-newton')
    results = solver.solve(m)
    predicted_x = 4.90547401
    self.assertAlmostEqual(predicted_x, m.x.value)
    self.assertEqual(results.problem.number_of_constraints, 1)
    self.assertEqual(results.problem.number_of_variables, 1)
    self.assertEqual(results.problem.number_of_continuous_variables, 1)
    self.assertEqual(results.problem.number_of_binary_variables, 0)
    self.assertEqual(results.problem.number_of_integer_variables, 0)
    self.assertGreater(results.solver.wallclock_time, 0.0)
    self.assertEqual(results.solver.termination_condition, pyo.TerminationCondition.feasible)
    self.assertEqual(results.solver.status, pyo.SolverStatus.ok)
    self.assertGreater(results.solver.number_of_function_evaluations, 0)