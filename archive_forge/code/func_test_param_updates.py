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
@parameterized.expand(input=all_solvers)
def test_param_updates(self, name: str, opt_class: Type[SolverBase]):
    opt = pe.SolverFactory(name + '_v2')
    if not opt.available(exception_flag=False):
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.a1 = pe.Param(mutable=True)
    m.a2 = pe.Param(mutable=True)
    m.b1 = pe.Param(mutable=True)
    m.b2 = pe.Param(mutable=True)
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
    m.c2 = pe.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
    for a1, a2, b1, b2 in params_to_test:
        m.a1.value = a1
        m.a2.value = a2
        m.b1.value = b1
        m.b2.value = b2
        res = opt.solve(m)
        pe.assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        self.assertAlmostEqual(m.dual[m.c1], 1 + a1 / (a2 - a1))
        self.assertAlmostEqual(m.dual[m.c2], a1 / (a2 - a1))