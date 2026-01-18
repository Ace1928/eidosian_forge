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
def test_results_infeasible(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
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
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=m.y >= m.x)
    m.c2 = pe.Constraint(expr=m.y <= m.x - 1)
    with self.assertRaises(Exception):
        res = opt.solve(m)
    opt.config.load_solutions = False
    opt.config.raise_exception_on_nonoptimal_result = False
    res = opt.solve(m)
    self.assertNotEqual(res.solution_status, SolutionStatus.optimal)
    if isinstance(opt, Ipopt):
        acceptable_termination_conditions = {TerminationCondition.locallyInfeasible, TerminationCondition.unbounded}
    else:
        acceptable_termination_conditions = {TerminationCondition.provenInfeasible, TerminationCondition.infeasibleOrUnbounded}
    self.assertIn(res.termination_condition, acceptable_termination_conditions)
    self.assertAlmostEqual(m.x.value, None)
    self.assertAlmostEqual(m.y.value, None)
    self.assertTrue(res.incumbent_objective is None)
    if not isinstance(opt, Ipopt):
        with self.assertRaisesRegex(RuntimeError, '.*does not currently have a valid solution.*'):
            res.solution_loader.load_vars()
        with self.assertRaisesRegex(RuntimeError, '.*does not currently have valid duals.*'):
            res.solution_loader.get_duals()
        with self.assertRaisesRegex(RuntimeError, '.*does not currently have valid reduced costs.*'):
            res.solution_loader.get_reduced_costs()