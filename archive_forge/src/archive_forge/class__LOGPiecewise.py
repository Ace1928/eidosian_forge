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
class _LOGPiecewise(object):
    """
    Called to generate Piecewise constraint using the LOG formulation
    """

    def _Branching_Scheme(self, n):
        """
        Branching scheme for LOG, requires a gray code
        """
        BIGL = 2 ** n
        S = range(1, n + 1)
        G = {k: v for k, v in enumerate(_GrayCode(n), start=1)}
        L = {s: [k + 1 for k in range(BIGL + 1) if (k == 0 or G[k][s - 1] == 1) and (k == BIGL or G[k + 1][s - 1] == 1)] for s in S}
        R = {s: [k + 1 for k in range(BIGL + 1) if (k == 0 or G[k][s - 1] == 0) and (k == BIGL or G[k + 1][s - 1] == 0)] for s in S}
        return (S, L, R)

    def construct(self, pblock, x_var, y_var):
        if not _isPowerOfTwo(len(pblock._domain_pts) - 1):
            msg = "'%s' does not have a list of domain points with length (2^n)+1"
            raise ValueError(msg % (pblock.name,))
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_LOGPiecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        L_i = int(math.log(len_x_pts - 1, 2))
        S_i, B_LEFT, B_RIGHT = self._Branching_Scheme(L_i)
        polytopes = range(1, len_x_pts)
        vertices = range(1, len_x_pts + 1)
        bin_y_index = S_i
        pblock.LOG_lambda = Var(vertices, within=NonNegativeReals)
        lmda = pblock.LOG_lambda
        pblock.LOG_bin_y = Var(bin_y_index, within=Binary)
        bin_y = pblock.LOG_bin_y
        pblock.LOG_constraint1 = Constraint(expr=x_var == sum((lmda[v] * x_pts[v - 1] for v in vertices)))
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
            raise ValueError('Invalid Bound for _LOGPiecewise object')
        pblock.LOG_constraint2 = Constraint(expr=expr)
        pblock.LOG_constraint3 = Constraint(expr=sum((lmda[v] for v in vertices)) == 1)

        def con4_rule(model, s):
            return sum((lmda[v] for v in B_LEFT[s])) <= bin_y[s]
        pblock.LOG_constraint4 = Constraint(bin_y_index, rule=con4_rule)

        def con5_rule(model, s):
            return sum((lmda[v] for v in B_RIGHT[s])) <= 1 - bin_y[s]
        pblock.LOG_constraint5 = Constraint(bin_y_index, rule=con5_rule)