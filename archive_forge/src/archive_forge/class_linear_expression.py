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
class linear_expression(mutable_expression):
    """Context manager for mutable linear sums.

    This context manager is used to compute a linear sum while
    treating the summation as a mutable object.

    Note
    ----

    The preferred context manager is :py:class:`mutable_expression`.
    :py:class:`linear_expression` is an alias to
    :py:class:`mutable_expression` provided for backwards compatibility.

    """