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
class _DLOGPiecewise(object):
    """
    Called to generate Piecewise constraint using the DLOG formulation
    """

    def _Branching_Scheme(self, L):
        """
        Branching scheme for DLOG
        """
        MAX = 2 ** L
        mylists1 = {}
        for i in range(1, L + 1):
            mylists1[i] = []
            start = 1
            step = int(MAX / 2 ** i)
            while start < MAX:
                mylists1[i].extend([j for j in range(start, start + step)])
                start += 2 * step
        biglist = range(1, MAX + 1)
        mylists2 = {}
        for i in sorted(mylists1.keys()):
            mylists2[i] = []
            for j in biglist:
                if j not in mylists1[i]:
                    mylists2[i].append(j)
            mylists2[i] = sorted(mylists2[i])
        return (mylists1, mylists2)

    def construct(self, pblock, x_var, y_var):
        if not _isPowerOfTwo(len(pblock._domain_pts) - 1):
            msg = "'%s' does not have a list of domain points with length (2^n)+1"
            raise ValueError(msg % (pblock.name,))
        x_pts = pblock._domain_pts
        y_pts = pblock._range_pts
        bound_type = pblock._bound_type
        if None in [x_pts, y_pts, bound_type]:
            raise RuntimeError('_DLOGPiecewise: construct() called during invalid state.')
        len_x_pts = len(x_pts)
        L_i = int(math.log(len_x_pts - 1, 2))
        B_ZERO, B_ONE = self._Branching_Scheme(L_i)
        polytopes = range(1, len_x_pts)
        vertices = range(1, len_x_pts + 1)
        bin_y_index = range(1, L_i + 1)

        def polytope_verts(p):
            return range(p, p + 2)
        pblock.DLOG_lambda = Var(polytopes, vertices, within=PositiveReals)
        lmda = pblock.DLOG_lambda
        pblock.DLOG_bin_y = Var(bin_y_index, within=Binary)
        bin_y = pblock.DLOG_bin_y
        pblock.DLOG_constraint1 = Constraint(expr=x_var == sum((lmda[p, v] * x_pts[v - 1] for p in polytopes for v in polytope_verts(p))))
        LHS = y_var
        RHS = sum((lmda[p, v] * y_pts[v - 1] for p in polytopes for v in polytope_verts(p)))
        expr = None
        if bound_type == Bound.Upper:
            expr = LHS <= RHS
        elif bound_type == Bound.Lower:
            expr = LHS >= RHS
        elif bound_type == Bound.Equal:
            expr = LHS == RHS
        else:
            raise ValueError('Invalid Bound for _DLOGPiecewise object')
        pblock.DLOG_constraint2 = Constraint(expr=expr)
        pblock.DLOG_constraint3 = Constraint(expr=sum((lmda[p, v] for p in polytopes for v in polytope_verts(p))) == 1)

        def con4_rule(model, l):
            return sum((lmda[p, v] for p in B_ZERO[l] for v in polytope_verts(p))) <= bin_y[l]
        pblock.DLOG_constraint4 = Constraint(bin_y_index, rule=con4_rule)

        def con5_rule(model, l):
            return sum((lmda[p, v] for p in B_ONE[l] for v in polytope_verts(p))) <= 1 - bin_y[l]
        pblock.DLOG_constraint5 = Constraint(bin_y_index, rule=con5_rule)