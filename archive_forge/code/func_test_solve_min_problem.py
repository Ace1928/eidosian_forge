import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
from pyomo.opt import WriterFactory, SolverFactory
def test_solve_min_problem(self):
    m = ConcreteModel()
    m.x = Var([1, 2, 3], bounds=(4, 6), domain=Integers)
    m.y = Var(within=[1, 2, 3])
    m.c1 = Constraint(expr=m.y >= 2.5)

    @m.Constraint([1, 2, 3])
    def x_bounds(m, i):
        return m.x[i] >= 3 * (i - 1)
    m.obj = Objective(expr=m.x[m.y])
    results = SolverFactory('cp_optimizer').solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(value(m.x[3]), 6)
    self.assertEqual(value(m.y), 3)
    self.assertEqual(results.problem.number_of_objectives, 1)
    self.assertEqual(results.problem.sense, minimize)
    self.assertEqual(results.problem.lower_bound, 6)
    self.assertEqual(results.problem.upper_bound, 6)