from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
def test_reverse_numeric_differentiation_in_LOA(self):
    m = ConcreteModel()
    m.s = RangeSet(1300)
    m.x = Var(m.s, bounds=(-10, 10))
    m.d1 = Disjunct()
    m.d1.hypersphere = Constraint(expr=sum((m.x[i] ** 2 for i in m.s)) <= 1)
    m.d2 = Disjunct()
    m.d2.translated_hyper_sphere = Constraint(expr=sum(((m.x[i] - i) ** 2 for i in m.s)) <= 1)
    m.disjunction = Disjunction(expr=[m.d1, m.d2])
    m.obj = Objective(expr=sum((m.x[i] for i in m.s)))
    results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertTrue(value(m.d1.indicator_var))
    self.assertFalse(value(m.d2.indicator_var))
    x_val = -sqrt(1300) / 1300
    for x in m.x.values():
        self.assertAlmostEqual(value(x), x_val)
    self.assertAlmostEqual(results.problem.upper_bound, 1300 * x_val, places=6)