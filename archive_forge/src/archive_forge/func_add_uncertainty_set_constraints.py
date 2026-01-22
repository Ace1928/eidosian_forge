from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver
from pyomo.contrib.pyros.solve_data import (
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, is_certain_parameter
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.common.errors import ApplicationError
from pyomo.contrib.pyros.util import ABS_CON_CHECK_FEAS_TOL
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import (
import os
from copy import deepcopy
from itertools import product
def add_uncertainty_set_constraints(model, config):
    """
    Add inequality constraint(s) representing the uncertainty set.
    """
    model.util.uncertainty_set_constraint = config.uncertainty_set.set_as_constraint(uncertain_params=model.util.uncertain_param_vars, model=model, config=config)
    config.uncertainty_set.add_bounds_on_uncertain_parameters(model=model, config=config)
    uncertain_params = config.uncertain_params
    for i in range(len(uncertain_params)):
        if is_certain_parameter(uncertain_param_index=i, config=config):
            model.util.uncertain_param_vars[i].fix(config.nominal_uncertain_param_vals[i])
    return