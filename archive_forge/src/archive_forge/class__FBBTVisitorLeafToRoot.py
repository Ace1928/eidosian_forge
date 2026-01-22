from collections import defaultdict
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import (
from pyomo.common.numeric_types import native_types
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
class _FBBTVisitorLeafToRoot(StreamBasedExpressionVisitor):
    """
    This walker propagates bounds from the variables to each node in
    the expression tree (all the way to the root node).
    """

    def __init__(self, bnds_dict, integer_tol=0.0001, feasibility_tol=1e-08, ignore_fixed=False):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        integer_tol: float
        feasibility_tol: float
            If the bounds computed on the body of a constraint violate the bounds of
            the constraint by more than feasibility_tol, then the constraint is
            considered infeasible and an exception is raised. This tolerance is also
            used when performing certain interval arithmetic operations to ensure that
            none of the feasible region is removed due to floating point arithmetic and
            to prevent math domain errors (a larger value is more conservative).
        """
        super().__init__()
        self.bnds_dict = bnds_dict
        self.integer_tol = integer_tol
        self.feasibility_tol = feasibility_tol
        self.ignore_fixed = ignore_fixed

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return (False, result)
        return (True, expr)

    def beforeChild(self, node, child, child_idx):
        return _before_child_handlers[child.__class__](self, child)

    def exitNode(self, node, data):
        _prop_bnds_leaf_to_root_map[node.__class__](self, node, *node.args)