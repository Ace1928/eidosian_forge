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
class _SOS2Piecewise(object):
    """
    Called to generate Piecewise constraint using the SOS2 formulation
    """

    def construct(self, pblock, x_var, y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_SOS2Piecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        sos2_index = range(len_x_pts)
        sos2_y = pblock.SOS2_y = Var(sos2_index, within=NonNegativeReals)
        conlist = pblock.SOS2_constraint = ConstraintList()
        conlist.add((x_var - sum((sos2_y[i] * x_pts[i] for i in sos2_index)), 0))
        LHS = y_var
        RHS = sum((sos2_y[i] * y_pts[i] for i in sos2_index))
        expr = None
        if bound_type == Bound.Upper:
            conlist.add((None, LHS - RHS, 0))
        elif bound_type == Bound.Lower:
            conlist.add((0, LHS - RHS, None))
        elif bound_type == Bound.Equal:
            conlist.add((LHS - RHS, 0))
        else:
            raise ValueError('Invalid Bound for _SOS2Piecewise object')
        conlist.add((sum((sos2_y[j] for j in sos2_index)), 1))

        def SOS2_rule(model):
            return [sos2_y[i] for i in sos2_index]
        pblock.SOS2_sosconstraint = SOSConstraint(initialize=SOS2_rule, sos=2)