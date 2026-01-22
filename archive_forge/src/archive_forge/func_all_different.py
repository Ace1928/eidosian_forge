import types
from itertools import islice
import logging
import traceback
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.deprecation import (
from .numvalue import (
from .base import ExpressionBase
from .boolean_value import BooleanValue, BooleanConstant
from .expr_common import _and, _or, _equiv, _inv, _xor, _impl, ExpressionType
from .numeric_expr import NumericExpression
import operator
def all_different(*args):
    """Creates a new AllDifferentExpression

    Requires all of the arguments to take on a different value

    Usage: all_different(m.X1, m.X2, ...)
    """
    return AllDifferentExpression(list(_flattened_numeric_args(args)))