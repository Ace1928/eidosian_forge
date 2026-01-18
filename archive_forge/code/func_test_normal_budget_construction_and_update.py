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
def test_normal_budget_construction_and_update(self):
    """
        Test BudgetSet constructor and attribute setters work
        appropriately.
        """
    budget_mat = [[1, 0, 1], [0, 1, 0]]
    budget_rhs_vec = [1, 3]
    buset = BudgetSet(budget_mat, budget_rhs_vec)
    np.testing.assert_allclose(budget_mat, buset.budget_membership_mat)
    np.testing.assert_allclose(budget_rhs_vec, buset.budget_rhs_vec)
    np.testing.assert_allclose([[1, 0, 1], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], buset.coefficients_mat)
    np.testing.assert_allclose([1, 3, 0, 0, 0], buset.rhs_vec)
    np.testing.assert_allclose(np.zeros(3), buset.origin)
    buset.budget_membership_mat = [[1, 1, 0], [0, 0, 1]]
    buset.budget_rhs_vec = [3, 4]
    np.testing.assert_allclose([[1, 1, 0], [0, 0, 1]], buset.budget_membership_mat)
    np.testing.assert_allclose([3, 4], buset.budget_rhs_vec)
    np.testing.assert_allclose([[1, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], buset.coefficients_mat)
    np.testing.assert_allclose([3, 4, 0, 0, 0], buset.rhs_vec)
    buset.origin = [1, 0, -1.5]
    np.testing.assert_allclose([1, 0, -1.5], buset.origin)