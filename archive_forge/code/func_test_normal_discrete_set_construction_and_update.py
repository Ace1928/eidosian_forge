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
def test_normal_discrete_set_construction_and_update(self):
    """
        Test DiscreteScenarioSet constructor and setter work normally
        when scenarios are appropriate.
        """
    scenarios = [[0, 0, 0], [1, 2, 3]]
    dset = DiscreteScenarioSet(scenarios)
    np.testing.assert_allclose(scenarios, dset.scenarios, err_msg='BoxSet bounds not as expected')
    new_scenarios = [[0, 1, 2], [1, 2, 0], [3, 5, 4]]
    dset.scenarios = new_scenarios
    np.testing.assert_allclose(new_scenarios, dset.scenarios, err_msg='BoxSet bounds not as expected')