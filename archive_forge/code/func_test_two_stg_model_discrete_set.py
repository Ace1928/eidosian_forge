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
@unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
def test_two_stg_model_discrete_set(self):
    """
        Test PyROS successfully solves two-stage model with
        multiple scenarios.
        """
    m = ConcreteModel()
    m.x1 = Var(bounds=(0, 10))
    m.x2 = Var(bounds=(0, 10))
    m.u = Param(mutable=True, initialize=1.125)
    m.con = Constraint(expr=sqrt(m.u) * m.x1 - m.u * m.x2 <= 2)
    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u) ** 2)
    discrete_set = DiscreteScenarioSet(scenarios=[[0.25], [1.125], [2]])
    global_solver = SolverFactory('baron')
    pyros_solver = SolverFactory('pyros')
    res = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u], uncertainty_set=discrete_set, local_solver=global_solver, global_solver=global_solver, decision_rule_order=0, solve_master_globally=True, objective_focus=ObjectiveType.worst_case)
    self.assertEqual(res.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg='Failed to solve discrete set multiple scenarios instance to robust optimality')