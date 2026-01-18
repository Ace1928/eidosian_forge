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
def test_two_stg_mod_with_intersection_set(self):
    """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
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
    bset = BoxSet(bounds=[[1, 2], [0.5, 1.5]])
    iset = IntersectionSet(ellipsoid=ellipsoid, bset=bset)
    pyros_solver = SolverFactory('pyros')
    local_subsolver = SolverFactory('baron')
    global_subsolver = SolverFactory('baron')
    results = pyros_solver.solve(model=m, first_stage_variables=[m.x1, m.x2], second_stage_variables=[], uncertain_params=[m.u1, m.u2], uncertainty_set=iset, local_solver=local_subsolver, global_solver=global_subsolver, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': True})
    self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg='Did not identify robust optimal solution to problem instance.')
    self.assertGreater(results.iterations, 0, msg='Robust infeasible model terminated in 0 iterations (nominal case).')