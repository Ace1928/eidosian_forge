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
class testMasterFeasibilityUnitConsistency(unittest.TestCase):
    """
    Test cases for models with unit-laden model components.
    """

    @unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
    @unittest.skipUnless(baron_version < (23, 1, 5), 'Test known to fail beginning with Baron 23.1.5')
    def test_two_stg_mod_with_axis_aligned_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        from pyomo.environ import units as u
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None), units=u.m)
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u1 = Param(initialize=1.125, mutable=True, units=u.s)
        m.u2 = Param(initialize=1, mutable=True, units=u.m ** 2)
        m.con1 = Constraint(expr=m.x1 * m.u1 ** 0.5 - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u1 == m.x3)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])
        pyros_solver = SolverFactory('pyros')
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory('baron')
        results = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u1, m.u2], uncertainty_set=ellipsoid, local_solver=local_subsolver, global_solver=global_subsolver, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': True})
        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg='Did not identify robust optimal solution to problem instance.')
        self.assertGreater(results.iterations, 1, msg='PyROS requires no more than one iteration to solve the model. Hence master feasibility problem construction not tested. Consider implementing a more challenging model for this test case.')