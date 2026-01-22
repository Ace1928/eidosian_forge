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
class EquivalenceExpression(BinaryBooleanExpression):
    """
    Logical equivalence statement: Y_1 iff Y_2.

    """
    __slots__ = ()
    PRECEDENCE = 6

    def getname(self, *arg, **kwd):
        return 'iff'

    def _to_string(self, values, verbose, smap):
        return ' iff '.join(values)

    def _apply_operation(self, result):
        return result[0] == result[1]