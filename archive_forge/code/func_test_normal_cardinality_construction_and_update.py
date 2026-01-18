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
def test_normal_cardinality_construction_and_update(self):
    """
        Test CardinalitySet constructor and setter work normally
        when bounds are appropriate.
        """
    cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 3], gamma=2)
    np.testing.assert_allclose(cset.origin, [0, 0])
    np.testing.assert_allclose(cset.positive_deviation, [1, 3])
    np.testing.assert_allclose(cset.gamma, 2)
    self.assertEqual(cset.dim, 2)
    cset.origin = [1, 2]
    cset.positive_deviation = [3, 0]
    cset.gamma = 0.5
    np.testing.assert_allclose(cset.origin, [1, 2])
    np.testing.assert_allclose(cset.positive_deviation, [3, 0])
    np.testing.assert_allclose(cset.gamma, 0.5)