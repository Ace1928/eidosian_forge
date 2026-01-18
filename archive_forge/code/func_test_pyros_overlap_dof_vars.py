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
def test_pyros_overlap_dof_vars(self):
    """
        Test PyROS solver raises exception raised if there are Vars
        passed as both first-stage and second-stage.
        """
    mdl = self.build_simple_test_model()
    pyros = SolverFactory('pyros')
    local_solver = SimpleTestSolver()
    global_solver = SimpleTestSolver()
    exc_str = 'Arguments `first_stage_variables` and `second_stage_variables` contain at least one common Var object.'
    with LoggingIntercept(level=logging.ERROR) as LOG:
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(model=mdl, first_stage_variables=[mdl.x1], second_stage_variables=[mdl.x1, mdl.x2], uncertain_params=[mdl.u], uncertainty_set=BoxSet([[1 / 4, 2]]), local_solver=local_solver, global_solver=global_solver)
    log_msgs = LOG.getvalue().split('\n')[:-1]
    self.assertEqual(len(log_msgs), 3, 'Error message does not contain expected number of lines.')
    self.assertRegex(text=log_msgs[0], expected_regex='The following Vars were found in both `first_stage_variables`and `second_stage_variables`.*')
    self.assertRegex(text=log_msgs[1], expected_regex=" 'x1'")
    self.assertRegex(text=log_msgs[2], expected_regex='Ensure no Vars are included in both arguments.')