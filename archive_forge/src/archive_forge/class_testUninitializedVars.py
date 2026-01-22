import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
@unittest.skipUnless(baron_available and baron_license_is_valid, 'Global NLP solver is not available and licensed.')
class testUninitializedVars(unittest.TestCase):

    def test_uninitialized_vars(self):
        """
        Test a simple PyROS model instance with uninitialized
        first-stage and second-stage variables.
        """
        m = ConcreteModel()
        m.ell0 = Param(initialize=1)
        m.u0 = Param(initialize=3)
        m.ell = Param(initialize=1)
        m.u = Param(initialize=5)
        m.p = Param(initialize=m.u0, mutable=True)
        m.r = Param(initialize=0.1)
        m.x = Var(bounds=(m.ell0, m.u0))
        m.z = Var(bounds=(m.ell0, m.p))
        m.t = Var(initialize=1, bounds=(0, m.r))
        m.w = Var(bounds=(0, 1))
        m.obj = Objective(expr=-m.x ** 2 + m.z ** 2)
        m.t_lb_con = Constraint(expr=m.x - m.z <= m.t)
        m.t_ub_con = Constraint(expr=-m.t <= m.x - m.z)
        m.con1 = Constraint(expr=m.x - m.z >= 0.1)
        m.eq_con = Constraint(expr=m.w == 0.5 * m.t)
        box_set = BoxSet(bounds=((value(m.ell), value(m.u)),))
        local_solver = SolverFactory('ipopt')
        global_solver = SolverFactory('baron')
        pyros_solver = SolverFactory('pyros')
        for dr_order in [0, 1, 2]:
            model = m.clone()
            fsv = [model.x]
            ssv = [model.z, model.t]
            uncertain_params = [model.p]
            res = pyros_solver.solve(model=model, first_stage_variables=fsv, second_stage_variables=ssv, uncertain_params=uncertain_params, uncertainty_set=box_set, local_solver=local_solver, global_solver=global_solver, objective_focus=ObjectiveType.worst_case, decision_rule_order=2, solve_master_globally=True)
            self.assertEqual(res.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg=f'Returned termination condition for solve withdecision rule order {dr_order} is not return robust_optimal.')