import copy
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.base.expression import Expression
from . import linear
from .linear import _merge_dict, to_expression
Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a QuadraticRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        