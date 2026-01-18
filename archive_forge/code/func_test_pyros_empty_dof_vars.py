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
def test_pyros_empty_dof_vars(self):
    """
        Test PyROS solver raises exception raised if there are no
        first-stage variables or second-stage variables.
        """
    mdl = self.build_simple_test_model()
    pyros = SolverFactory('pyros')
    local_solver = SimpleTestSolver()
    global_solver = SimpleTestSolver()
    exc_str = 'Arguments `first_stage_variables` and `second_stage_variables` are both empty lists.'
    with self.assertRaisesRegex(ValueError, exc_str):
        pyros.solve(model=mdl, first_stage_variables=[], second_stage_variables=[], uncertain_params=[mdl.u], uncertainty_set=BoxSet([[1 / 4, 2]]), local_solver=local_solver, global_solver=global_solver)