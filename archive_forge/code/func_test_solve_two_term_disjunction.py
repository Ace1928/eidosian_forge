import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
def test_solve_two_term_disjunction(self):
    m = models.makeTwoTermDisj()
    m.obj = Objective(expr=m.x, sense=maximize)
    results = SolverFactory('gdpopt.enumerate').solve(m)
    self.assertEqual(results.solver.iterations, 2)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(results.problem.lower_bound, 9)
    self.assertEqual(results.problem.upper_bound, 9)
    self.assertEqual(value(m.x), 9)
    self.assertTrue(value(m.d[0].indicator_var))
    self.assertFalse(value(m.d[1].indicator_var))