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
def test_normal_construction_and_update(self):
    """
        Test IntersectionSet constructor and setter
        work normally when arguments are appropriate.
        """
    bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
    aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])
    iset = IntersectionSet(box_set=bset, axis_aligned_set=aset)
    self.assertIn(bset, iset.all_sets, msg="IntersectionSet 'all_sets' attribute does notcontain expected BoxSet")
    self.assertIn(aset, iset.all_sets, msg="IntersectionSet 'all_sets' attribute does notcontain expected AxisAlignedEllipsoidalSet")