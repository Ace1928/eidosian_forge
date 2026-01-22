import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
@unittest.skipUnless(AmplInterface.available(), 'AmplInterface is not available')
class TestSecantNewtonPyomo(unittest.TestCase):

    def test_available(self):
        solver = pyo.SolverFactory('scipy.secant-newton')
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())
        sp_version = tuple((int(num) for num in scipy.__version__.split('.')))
        self.assertEqual(sp_version, solver.version())

    def test_solve(self):
        m, _ = make_scalar_model()
        solver = pyo.SolverFactory('scipy.secant-newton')
        results = solver.solve(m)
        predicted_x = 4.90547401
        self.assertAlmostEqual(predicted_x, m.x.value)
        self.assertFalse(solver.converged_with_secant())

    def test_solve_doesnt_converge(self):
        m, _ = make_scalar_model()
        m.x.set_value(30000000000.0)
        solver = pyo.SolverFactory('scipy.secant-newton')
        with self.assertRaisesRegex(RuntimeError, 'Failed to converge'):
            results = solver.solve(m)
        self.assertFalse(solver.converged_with_secant())

    def test_too_many_iter(self):
        m, _ = make_scalar_model()
        solver = pyo.SolverFactory('scipy.secant-newton')
        solver.set_options({'newton_iter': 5})
        with self.assertRaisesRegex(RuntimeError, 'Failed to converge'):
            results = solver.solve(m)

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

    def test_results_object_without_full_output(self):
        m, _ = make_scalar_model()
        solver = pyo.SolverFactory('scipy.secant-newton')
        solver.set_options(dict(full_output=False))
        results = solver.solve(m)
        predicted_x = 4.90547401
        self.assertAlmostEqual(predicted_x, m.x.value)
        self.assertEqual(results.problem.number_of_constraints, 1)
        self.assertEqual(results.problem.number_of_variables, 1)
        self.assertEqual(results.problem.number_of_continuous_variables, 1)
        self.assertEqual(results.problem.number_of_binary_variables, 0)
        self.assertEqual(results.problem.number_of_integer_variables, 0)
        self.assertGreater(results.solver.wallclock_time, 0.0)
        self.assertIs(results.solver.termination_condition, pyo.TerminationCondition.unknown)
        with self.assertRaises(AttributeError):
            n_eval = results.solver.number_of_function_evaluations

    def test_solve_linear(self):
        m, _ = make_linear_scalar_model()
        solver = pyo.SolverFactory('scipy.secant-newton')
        results = solver.solve(m)
        self.assertAlmostEqual(m.x.value, 30.1 / 12.5)
        self.assertTrue(solver.converged_with_secant())