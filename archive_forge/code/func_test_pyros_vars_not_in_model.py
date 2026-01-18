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
def test_pyros_vars_not_in_model(self):
    """
        Test PyROS appropriately raises exception if there are
        variables not included in active model objective
        or constraints which are not descended from model.
        """
    mdl = self.build_simple_test_model()
    mdl.name = 'model1'
    mdl2 = self.build_simple_test_model()
    mdl2.name = 'model2'
    local_solver = SimpleTestSolver()
    global_solver = SimpleTestSolver()
    pyros = SolverFactory('pyros')
    mdl.bad_con = Constraint(expr=mdl2.x1 + mdl2.x2 >= 1)
    desc_dof_map = [('first-stage', [mdl2.x1], [], 2), ('second-stage', [], [mdl2.x2], 2), ('state', [mdl.x1], [], 3)]
    for vardesc, first_stage_vars, second_stage_vars, numlines in desc_dof_map:
        with LoggingIntercept(level=logging.ERROR) as LOG:
            exc_str = f'Found entries of {vardesc} variables not descended from.*model.*'
            with self.assertRaisesRegex(ValueError, exc_str):
                pyros.solve(model=mdl, first_stage_variables=first_stage_vars, second_stage_variables=second_stage_vars, uncertain_params=[mdl.u], uncertainty_set=BoxSet([[1 / 4, 2]]), local_solver=local_solver, global_solver=global_solver)
        log_msgs = LOG.getvalue().split('\n')[:-1]
        self.assertEqual(len(log_msgs), numlines, 'Error-level log message does not contain expected number of lines.')
        self.assertRegex(text=log_msgs[0], expected_regex=f"The following {vardesc} variables.*not descended from.*model with name 'model1'")