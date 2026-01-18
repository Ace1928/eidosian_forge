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
def test_polyhedral_set_as_constraint(self):
    """
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        """
    A = [[1, 0], [0, 1]]
    b = [0, 0]
    m = ConcreteModel()
    m.p1 = Var(initialize=0)
    m.p2 = Var(initialize=0)
    polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
    m.uncertainty_set_constr = polyhedral_set.set_as_constraint(uncertain_params=[m.p1, m.p2])
    self.assertEqual(len(A), len(m.uncertainty_set_constr.index_set()), msg='Polyhedral uncertainty set constraints must be as many as thenumber of rows in the matrix A.')