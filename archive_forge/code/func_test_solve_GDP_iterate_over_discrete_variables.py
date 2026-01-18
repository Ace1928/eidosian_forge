import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
def test_solve_GDP_iterate_over_discrete_variables(self):
    m = models.makeTwoTermDisj()
    self.modify_two_term_disjunction(m)
    results = SolverFactory('gdpopt.enumerate').solve(m, force_subproblem_nlp=True)
    self.assertEqual(results.solver.iterations, 6)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(results.problem.lower_bound, -11)
    self.assertEqual(results.problem.upper_bound, -11)
    self.assertEqual(value(m.x), 9)
    self.assertEqual(value(m.y), 2)
    self.assertTrue(value(m.d[0].indicator_var))
    self.assertFalse(value(m.d[1].indicator_var))