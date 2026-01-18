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
@unittest.skipUnless(ipopt_available, 'IPOPT is not available.')
@unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
def test_pyros_kwargs_with_overlap(self):
    """
        Test PyROS works as expected when there is overlap between
        keyword arguments passed explicitly and implicitly
        through `options`.
        """
    m = ConcreteModel()
    m.x1 = Var(initialize=0, bounds=(0, None))
    m.x2 = Var(initialize=0, bounds=(0, None))
    m.x3 = Var(initialize=0, bounds=(None, None))
    m.u1 = Param(initialize=1.125, mutable=True)
    m.u2 = Param(initialize=1, mutable=True)
    m.con1 = Constraint(expr=m.x1 * m.u1 ** 0.5 - m.x2 * m.u1 <= 2)
    m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u1 == m.x3)
    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)
    ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])
    pyros_solver = SolverFactory('pyros')
    local_subsolver = SolverFactory('ipopt')
    global_subsolver = SolverFactory('baron')
    results = pyros_solver.solve(model=m, first_stage_variables=[m.x1, m.x2], second_stage_variables=[], uncertain_params=[m.u1, m.u2], uncertainty_set=ellipsoid, local_solver=local_subsolver, global_solver=global_subsolver, bypass_local_separation=True, solve_master_globally=True, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': False, 'max_iter': 1, 'time_limit': 1000})
    self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.max_iter, msg='Termination condition not as expected')
    self.assertEqual(results.iterations, 1, msg='Number of iterations not as expected')
    config = results.config
    self.assertEqual(config.bypass_local_separation, True, msg='Resolved value of kwarg `bypass_local_separation` not as expected.')
    self.assertEqual(config.solve_master_globally, True, msg='Resolved value of kwarg `solve_master_globally` not as expected.')
    self.assertEqual(config.max_iter, 1, msg='Resolved value of kwarg `max_iter` not as expected.')
    self.assertEqual(config.objective_focus, ObjectiveType.worst_case, msg='Resolved value of kwarg `objective_focus` not as expected.')
    self.assertEqual(config.time_limit, 1000.0, msg='Resolved value of kwarg `time_limit` not as expected.')