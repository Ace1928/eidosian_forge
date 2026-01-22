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
class ImplicationExpression(BinaryBooleanExpression):
    """
    Logical Implication statement: Y_1 --> Y_2.
    """
    __slots__ = ()
    PRECEDENCE = 6

    def getname(self, *arg, **kwd):
        return 'implies'

    def _to_string(self, values, verbose, smap):
        return ' --> '.join(values)

    def _apply_operation(self, result):
        return not result[0] or result[1]