import logging
import math
import itertools
import operator
import types
import enum
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.util import flatten_tuple
def _find_M(self, x_pts, y_pts, bound_type):
    len_x_pts = len(x_pts)
    _self_M_func = self._M_func
    M_final = {}
    for j in range(1, len_x_pts):
        index = j
        if bound_type == Bound.Lower:
            M_final[index] = min([0.0, min([_self_M_func(x_pts[k], y_pts[k], x_pts[j - 1], y_pts[j - 1], x_pts[j], y_pts[j]) for k in range(len_x_pts)])])
        elif bound_type == Bound.Upper:
            M_final[index] = max([0.0, max([_self_M_func(x_pts[k], y_pts[k], x_pts[j - 1], y_pts[j - 1], x_pts[j], y_pts[j]) for k in range(len_x_pts)])])
        else:
            raise ValueError('Invalid Bound passed to _find_M function')
        if M_final[index] == 0.0:
            del M_final[index]
    return M_final