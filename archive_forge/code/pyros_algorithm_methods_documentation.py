from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, coefficient_matching
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import _VarData as VarData
from itertools import chain
from pyomo.common.dependencies import numpy as np

    GRCS algorithm implementation
    :model_data: ROSolveData object with deterministic model information
    :config: ConfigBlock for the instance being solved
    