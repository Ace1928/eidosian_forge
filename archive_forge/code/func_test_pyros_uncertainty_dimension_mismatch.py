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
def test_pyros_uncertainty_dimension_mismatch(self):
    """
        Test PyROS solver raises exception if uncertainty
        set dimension does not match the number
        of uncertain parameters.
        """
    mdl = self.build_simple_test_model()
    pyros = SolverFactory('pyros')
    local_solver = SimpleTestSolver()
    global_solver = SimpleTestSolver()
    exc_str = 'Length of argument `uncertain_params` does not match dimension of argument `uncertainty_set` \\(1 != 2\\).'
    with self.assertRaisesRegex(ValueError, exc_str):
        pyros.solve(model=mdl, first_stage_variables=[mdl.x1], second_stage_variables=[mdl.x2], uncertain_params=[mdl.u], uncertainty_set=BoxSet([[1 / 4, 2], [0, 1]]), local_solver=local_solver, global_solver=global_solver)