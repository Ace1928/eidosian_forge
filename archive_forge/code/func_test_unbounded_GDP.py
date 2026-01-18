import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
def test_unbounded_GDP(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-1, 10))
    m.y = Var(bounds=(2, 3))
    m.z = Var()
    m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
    m.o = Objective(expr=m.z)
    results = SolverFactory('gdpopt.enumerate').solve(m)
    self.assertEqual(results.solver.iterations, 1)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.unbounded)
    self.assertEqual(results.problem.lower_bound, -float('inf'))
    self.assertEqual(results.problem.upper_bound, -float('inf'))