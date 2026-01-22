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
class _CCPiecewise(object):
    """
    Called to generate Piecewise constraint using the CC formulation
    """

    def construct(self, pblock, x_var, y_var):
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_CCPiecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        polytopes = range(1, len_x_pts)
        vertices = range(1, len_x_pts + 1)

        def vertex_polys(v):
            if v == 1:
                return [v]
            if v == len_x_pts:
                return [v - 1]
            else:
                return [v - 1, v]
        pblock.CC_lambda = Var(vertices, within=NonNegativeReals)
        lmda = pblock.CC_lambda
        pblock.CC_bin_y = Var(polytopes, within=Binary)
        bin_y = pblock.CC_bin_y
        pblock.CC_constraint1 = Constraint(expr=x_var == sum((lmda[v] * x_pts[v - 1] for v in vertices)))
        LHS = y_var
        RHS = sum((lmda[v] * y_pts[v - 1] for v in vertices))
        expr = None
        if bound_type == Bound.Upper:
            expr = LHS <= RHS
        elif bound_type == Bound.Lower:
            expr = LHS >= RHS
        elif bound_type == Bound.Equal:
            expr = LHS == RHS
        else:
            raise ValueError('Invalid Bound for _CCPiecewise object')
        pblock.CC_constraint2 = Constraint(expr=expr)
        pblock.CC_constraint3 = Constraint(expr=sum((lmda[v] for v in vertices)) == 1)

        def con4_rule(model, v):
            return lmda[v] <= sum((bin_y[p] for p in vertex_polys(v)))
        pblock.CC_constraint4 = Constraint(vertices, rule=con4_rule)
        pblock.CC_constraint5 = Constraint(expr=sum((bin_y[p] for p in polytopes)) == 1)