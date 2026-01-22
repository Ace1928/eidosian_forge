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
def _prop_bnds_leaf_to_root_DivisionExpression(visitor, node, arg1, arg2):
    """

    Parameters
    ----------
    visitor: _FBBTVisitorLeafToRoot
    node: pyomo.core.expr.numeric_expr.DivisionExpression
    arg1: dividend
    arg2: divisor
    """
    bnds_dict = visitor.bnds_dict
    bnds_dict[node] = interval.div(*bnds_dict[arg1], *bnds_dict[arg2], feasibility_tol=visitor.feasibility_tol)