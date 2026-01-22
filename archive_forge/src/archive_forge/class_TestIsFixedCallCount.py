import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestIsFixedCallCount(unittest.TestCase):
    """Tests for PR#1402 (669e7b2b)"""

    def setup(self, skip_trivial_constraints):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c1 = Constraint(expr=m.x + m.y == 1)
        m.c2 = Constraint(expr=m.x <= 1)
        self.assertFalse(m.c2.has_lb())
        self.assertTrue(m.c2.has_ub())
        self._model = m
        self._opt = SolverFactory('cplex_persistent')
        self._opt.set_instance(self._model, skip_trivial_constraints=skip_trivial_constraints)

    def test_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=True)
        self._model.x.fix(1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())
        with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_skip_trivial_and_call_count_for_unfixed_con_is_two(self):
        self.setup(skip_trivial_constraints=True)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())
        with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 2)

    def test_skip_trivial_and_call_count_for_unfixed_equality_con_is_three(self):
        self.setup(skip_trivial_constraints=True)
        self._model.c2 = Constraint(expr=self._model.x == 1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())
        with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 3)

    def test_dont_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self._model.x.fix(1)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())
        with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_dont_skip_trivial_and_call_count_for_unfixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())
        with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)