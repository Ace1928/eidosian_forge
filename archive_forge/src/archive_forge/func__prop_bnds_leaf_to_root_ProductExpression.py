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
def _prop_bnds_leaf_to_root_ProductExpression(visitor, node, arg1, arg2):
    """

    Parameters
    ----------
    visitor: _FBBTVisitorLeafToRoot
    node: pyomo.core.expr.numeric_expr.ProductExpression
    arg1: First arg in product expression
    arg2: Second arg in product expression
    """
    bnds_dict = visitor.bnds_dict
    if arg1 is arg2:
        bnds_dict[node] = interval.power(*bnds_dict[arg1], 2, 2, visitor.feasibility_tol)
    else:
        bnds_dict[node] = interval.mul(*bnds_dict[arg1], *bnds_dict[arg2])