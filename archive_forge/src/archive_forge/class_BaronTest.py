from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Objective, Var, log10, minimize
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
@unittest.skipIf(not baron_available, "The 'BARON' solver is not available")
class BaronTest(unittest.TestCase):
    """Test the BARON interface."""

    def test_log10(self):
        with SolverFactory('baron') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=log10(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    def test_abs(self):
        with SolverFactory('baron') as opt:
            m = ConcreteModel()
            m.x = Var(bounds=(-100, 1))
            m.c = Constraint(expr=abs(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    def test_pow(self):
        with SolverFactory('baron') as opt:
            m = ConcreteModel()
            m.x = Var(bounds=(10, 100))
            m.y = Var(bounds=(1, 10))
            m.c = Constraint(expr=m.x ** m.y >= 20)
            m.obj = Objective(expr=m.x, sense=minimize)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    def test_BARON_option_warnings(self):
        os = StringIO()
        with LoggingIntercept(os, 'pyomo.solvers'):
            m = ConcreteModel()
            m.x = Var()
            m.obj = Objective(expr=m.x ** 2)
            with SolverFactory('baron') as opt:
                results = opt.solve(m, options={'ResName': 'results.lst', 'TimName': 'results.tim'})
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertIn('Ignoring user-specified option "ResName=results.lst"', os.getvalue())
        self.assertIn('Ignoring user-specified option "TimName=results.tim"', os.getvalue())