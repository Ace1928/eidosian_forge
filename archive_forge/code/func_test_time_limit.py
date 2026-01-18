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
def test_time_limit(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    from sys import platform
    if platform == 'win32':
        raise unittest.SkipTest
    N = 30
    m = pe.ConcreteModel()
    m.jobs = pe.Set(initialize=list(range(N)))
    m.tasks = pe.Set(initialize=list(range(N)))
    m.x = pe.Var(m.jobs, m.tasks, bounds=(0, 1))
    random.seed(0)
    coefs = list()
    lin_vars = list()
    for j in m.jobs:
        for t in m.tasks:
            coefs.append(random.uniform(0, 10))
            lin_vars.append(m.x[j, t])
    obj_expr = LinearExpression(linear_coefs=coefs, linear_vars=lin_vars, constant=0)
    m.obj = pe.Objective(expr=obj_expr, sense=pe.maximize)
    m.c1 = pe.Constraint(m.jobs)
    m.c2 = pe.Constraint(m.tasks)
    for j in m.jobs:
        expr = LinearExpression(linear_coefs=[1] * N, linear_vars=[m.x[j, t] for t in m.tasks], constant=0)
        m.c1[j] = expr == 1
    for t in m.tasks:
        expr = LinearExpression(linear_coefs=[1] * N, linear_vars=[m.x[j, t] for j in m.jobs], constant=0)
        m.c2[t] = expr == 1
    if isinstance(opt, Ipopt):
        opt.config.time_limit = 1e-06
    else:
        opt.config.time_limit = 0
    opt.config.load_solutions = False
    opt.config.raise_exception_on_nonoptimal_result = False
    res = opt.solve(m)
    self.assertIn(res.termination_condition, {TerminationCondition.maxTimeLimit, TerminationCondition.iterationLimit})