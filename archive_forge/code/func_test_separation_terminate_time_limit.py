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
@unittest.skipUnless(SolverFactory('baron').license_is_valid(), 'Global NLP solver is not available and licensed.')
def test_separation_terminate_time_limit(self):
    """
        Test PyROS time limit status returned in event
        separation problem times out.
        """
    m = ConcreteModel()
    m.x1 = Var(initialize=0, bounds=(0, None))
    m.x2 = Var(initialize=0, bounds=(0, None))
    m.x3 = Var(initialize=0, bounds=(None, None))
    m.u = Param(initialize=1.125, mutable=True)
    m.con1 = Constraint(expr=m.x1 * m.u ** 0.5 - m.x2 * m.u <= 2)
    m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)
    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
    interval = BoxSet(bounds=[(0.25, 2)])
    pyros_solver = SolverFactory('pyros')
    local_subsolver = TimeDelaySolver(calls_to_sleep=0, sub_solver=SolverFactory('baron'), max_time=1)
    global_subsolver = SolverFactory('baron')
    results = pyros_solver.solve(model=m, first_stage_variables=[m.x1, m.x2], second_stage_variables=[], uncertain_params=[m.u], uncertainty_set=interval, local_solver=local_subsolver, global_solver=global_subsolver, objective_focus=ObjectiveType.worst_case, solve_master_globally=True, time_limit=1)
    self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.time_out, msg='Returned termination condition is not return time_out.')