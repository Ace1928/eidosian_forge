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
def test_type_error_on_invalid_arg(self):
    """
        Test TypeError raised if an argument not of type
        UncertaintySet is passed to the IntersectionSet
        constructor or appended to 'all_sets'.
        """
    bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
    aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])
    exc_str = "Entry '1' of the argument `all_sets` is not An `UncertaintySet` object.*\\(provided type 'int'\\)"
    with self.assertRaisesRegex(TypeError, exc_str):
        IntersectionSet(box_set=bset, axis_set=aset, invalid_arg=1)
    iset = IntersectionSet(box_set=bset, axis_set=aset)
    with self.assertRaisesRegex(TypeError, exc_str):
        iset.all_sets.append(1)