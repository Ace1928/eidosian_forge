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
@unittest.skipUnless(SolverFactory('gams:ipopt').available(exception_flag=False), 'Local NLP solver GAMS/IPOPT is not available.')
def test_pyros_gams_ipopt(self):
    """
        Test PyROS usage with solver GAMS ipopt
        works without exceptions.
        """
    m = self.simple_nlp_model()
    ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])
    pyros_solver = SolverFactory('pyros')
    local_subsolver = SolverFactory('gams:ipopt')
    global_subsolver = SolverFactory('gams:ipopt')
    results = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u1, m.u2], uncertainty_set=ellipsoid, local_solver=local_subsolver, global_solver=global_subsolver, objective_focus=ObjectiveType.worst_case, solve_master_globally=False, bypass_global_separation=True)
    self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_feasible, msg='Did not identify robust optimal solution to problem instance.')
    self.assertFalse(math.isnan(results.time), msg='PyROS solve time is nan (expected otherwise since subsolvertime estimates are made using TicTocTimer')