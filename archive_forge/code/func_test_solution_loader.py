import random
import math
from typing import Type
import pyomo.environ as pe
from pyomo import gdp
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus, Results
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.ipopt import Ipopt
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.core.expr.numeric_expr import LinearExpression
@parameterized.expand(input=_load_tests(all_solvers))
def test_solution_loader(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(1, None))
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=(0, m.y - m.x, None))
    m.c2 = pe.Constraint(expr=(0, m.y - m.x + 1, None))
    opt.config.load_solutions = False
    res = opt.solve(m)
    self.assertIsNone(m.x.value)
    self.assertIsNone(m.y.value)
    res.solution_loader.load_vars()
    self.assertAlmostEqual(m.x.value, 1)
    self.assertAlmostEqual(m.y.value, 1)
    m.x.value = None
    m.y.value = None
    res.solution_loader.load_vars([m.y])
    self.assertAlmostEqual(m.y.value, 1)
    primals = res.solution_loader.get_primals()
    self.assertIn(m.x, primals)
    self.assertIn(m.y, primals)
    self.assertAlmostEqual(primals[m.x], 1)
    self.assertAlmostEqual(primals[m.y], 1)
    primals = res.solution_loader.get_primals([m.y])
    self.assertNotIn(m.x, primals)
    self.assertIn(m.y, primals)
    self.assertAlmostEqual(primals[m.y], 1)
    reduced_costs = res.solution_loader.get_reduced_costs()
    self.assertIn(m.x, reduced_costs)
    self.assertIn(m.y, reduced_costs)
    self.assertAlmostEqual(reduced_costs[m.x], 1)
    self.assertAlmostEqual(reduced_costs[m.y], 0)
    reduced_costs = res.solution_loader.get_reduced_costs([m.y])
    self.assertNotIn(m.x, reduced_costs)
    self.assertIn(m.y, reduced_costs)
    self.assertAlmostEqual(reduced_costs[m.y], 0)
    duals = res.solution_loader.get_duals()
    self.assertIn(m.c1, duals)
    self.assertIn(m.c2, duals)
    self.assertAlmostEqual(duals[m.c1], 1)
    self.assertAlmostEqual(duals[m.c2], 0)
    duals = res.solution_loader.get_duals([m.c1])
    self.assertNotIn(m.c2, duals)
    self.assertIn(m.c1, duals)
    self.assertAlmostEqual(duals[m.c1], 1)