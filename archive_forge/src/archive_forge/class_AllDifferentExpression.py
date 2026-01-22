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
class AllDifferentExpression(NaryBooleanExpression):
    """
    Logical expression that all of the N child statements have different values.
    All arguments are expected to be discrete-valued.
    """
    __slots__ = ()
    PRECEDENCE = None

    def getname(self, *arg, **kwd):
        return 'all_different'

    def _to_string(self, values, verbose, smap):
        return 'all_different(%s)' % ', '.join(values)

    def _apply_operation(self, result):
        last = None
        for val in sorted(result):
            if last == val:
                return False
            last = val
        return True