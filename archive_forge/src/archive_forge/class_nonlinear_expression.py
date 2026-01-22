import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
class nonlinear_expression(mutable_expression):
    """Context manager for mutable nonlinear sums.

    This context manager is used to compute a general nonlinear sum
    while treating the summation as a mutable object.

    Note
    ----

    The preferred context manager is :py:class:`mutable_expression`, as
    the return type will be the most specific of
    :py:class:`SumExpression`, :py:class:`LinearExpression`, or
    :py:class:`NPV_SumExpression`.  This context manager will *always*
    return a :py:class:`SumExpression`.

    """

    def __enter__(self):
        self.e = _MutableSumExpression([])
        return self.e