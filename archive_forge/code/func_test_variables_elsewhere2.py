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
def test_variables_elsewhere2(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.z = pe.Var()
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=m.y >= m.x)
    m.c2 = pe.Constraint(expr=m.y >= -m.x)
    m.c3 = pe.Constraint(expr=m.y >= m.z + 1)
    m.c4 = pe.Constraint(expr=m.y >= -m.z + 1)
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    sol = res.solution_loader.get_primals()
    self.assertIn(m.x, sol)
    self.assertIn(m.y, sol)
    self.assertIn(m.z, sol)
    del m.c3
    del m.c4
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(res.incumbent_objective, 0)
    sol = res.solution_loader.get_primals()
    self.assertIn(m.x, sol)
    self.assertIn(m.y, sol)
    self.assertNotIn(m.z, sol)