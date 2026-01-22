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
class NotExpression(UnaryBooleanExpression):
    """
    This is the node for a NotExpression, this node should have exactly one child
    """
    PRECEDENCE = 2

    def getname(self, *arg, **kwd):
        return 'Logical Negation'

    def _to_string(self, values, verbose, smap):
        return '~%s' % values[0]

    def _apply_operation(self, result):
        return not result[0]