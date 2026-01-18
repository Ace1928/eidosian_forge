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
@unittest.skipUnless(SolverFactory(mip_solver).available(), 'MIP solver not available')
def test_solve_discrete_problem_unbounded(self):
    m = ConcreteModel()
    m.GDPopt_utils = Block()
    m.x = Var(bounds=(-1, 10))
    m.y = Var(bounds=(2, 3))
    m.z = Var()
    m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
    m.o = Objective(expr=m.z)
    m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
    m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0], m.d._autodisjuncts[1]]
    TransformationFactory('gdp.bigm').apply_to(m)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
        solver = SolverFactory('gdpopt.loa')
        dummy = Block()
        dummy.timing = Bunch()
        with time_code(dummy.timing, 'main', is_main_timer=True):
            tc = solve_MILP_discrete_problem(m.GDPopt_utils, dummy, solver.CONFIG(dict(mip_solver=mip_solver)))
        self.assertIn('Discrete problem was unbounded. Re-solving with arbitrary bound values', output.getvalue().strip())
    self.assertIs(tc, TerminationCondition.unbounded)