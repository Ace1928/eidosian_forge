import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
class CPLEXShellSolvePrioritiesFile(unittest.TestCase):
    """Integration test on the end-to-end application of priorities via the `Suffix` through a `solve()`"""

    def get_mock_model_with_priorities(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers)
        m.s = RangeSet(0, 9)
        m.y = Var(m.s, domain=Integers)
        m.o = Objective(expr=m.x + sum(m.y), sense=minimize)
        m.c = Constraint(expr=m.x >= 1)
        m.c2 = Constraint(expr=quicksum((m.y[i] for i in m.s)) >= 10)
        m.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        m.direction = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        m.priority.set_value(m.x, 1)
        m.priority.set_value(m.y, 2, expand=False)
        m.direction.set_value(m.y, BranchDirection.down, expand=True)
        m.direction.set_value(m.y[9], BranchDirection.up)
        return m

    def test_use_variable_priorities(self):
        model = self.get_mock_model_with_priorities()
        with SolverFactory('_mock_cplex') as opt:
            opt._presolve(model, priorities=True, keepfiles=True, symbolic_solver_labels=True)
            with open(opt._priorities_file_name, 'r') as ord_file:
                priorities_file = ord_file.read()
        self.assertEqual(priorities_file, '* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x 1\n DN y(0) 2\n DN y(1) 2\n DN y(2) 2\n DN y(3) 2\n DN y(4) 2\n DN y(5) 2\n DN y(6) 2\n DN y(7) 2\n DN y(8) 2\n UP y(9) 2\nENDATA\n')
        self.assertIn('read %s\n' % (opt._priorities_file_name,), opt._command.script)

    def test_ignore_variable_priorities(self):
        model = self.get_mock_model_with_priorities()
        with SolverFactory('_mock_cplex') as opt:
            opt._presolve(model, priorities=False, keepfiles=True)
            self.assertIsNone(opt._priorities_file_name)
            self.assertNotIn('.ord', opt._command.script)

    def test_can_use_manual_priorities_file_with_lp_solve(self):
        """Test that we can pass an LP file (not a pyomo model) along with a priorities file to `.solve()`"""
        model = self.get_mock_model_with_priorities()
        with SolverFactory('_mock_cplex') as pre_opt:
            pre_opt._presolve(model, priorities=True, keepfiles=True)
            lp_file = pre_opt._problem_files[0]
            priorities_file_name = pre_opt._priorities_file_name
            with open(priorities_file_name, 'r') as ord_file:
                provided_priorities_file = ord_file.read()
        with SolverFactory('_mock_cplex') as opt:
            opt._presolve(lp_file, priorities=True, priorities_file=priorities_file_name, keepfiles=True)
            self.assertIn('.ord', opt._command.script)
            with open(opt._priorities_file_name, 'r') as ord_file:
                priorities_file = ord_file.read()
        self.assertEqual(priorities_file, provided_priorities_file)