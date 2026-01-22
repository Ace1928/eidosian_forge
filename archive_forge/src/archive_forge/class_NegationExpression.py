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
class NegationExpression(NumericExpression):
    """
    Negation expressions::

        - x
    """
    __slots__ = ()
    PRECEDENCE = 4

    def nargs(self):
        return 1

    def getname(self, *args, **kwds):
        return 'neg'

    def _compute_polynomial_degree(self, result):
        return result[0]

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f'{self.getname()}({values[0]})'
        tmp = values[0]
        if tmp[0] == '-':
            return tmp[1:].strip()
        return '- ' + tmp

    def _apply_operation(self, result):
        return -result[0]

    def __neg__(self):
        return self._args_[0]