import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
def test_solve_GDP_do_not_iterate_over_Boolean_variables(self):
    m = models.makeLogicalConstraintsOnDisjuncts()
    results = SolverFactory('gdpopt.enumerate').solve(m)
    self.assertEqual(results.solver.iterations, 4)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(results.problem.lower_bound, 8)
    self.assertEqual(results.problem.upper_bound, 8)
    self.assertTrue(value(m.d[2].indicator_var))
    self.assertTrue(value(m.d[3].indicator_var))
    self.assertFalse(value(m.d[1].indicator_var))
    self.assertFalse(value(m.d[4].indicator_var))
    self.assertEqual(value(m.x), 8)
    self.assertNotEqual(value(m.Y[1]), value(m.Y[2]))