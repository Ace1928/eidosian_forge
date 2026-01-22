import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
def _set_obj_factor_if_necessary(self, obj_factor):
    if obj_factor != self._cached_obj_factor:
        self._pyomo_nlp.set_obj_factor(obj_factor)
        self._cached_obj_factor = obj_factor