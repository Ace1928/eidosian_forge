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
class ProductExpression(NumericExpression):
    """
    Product expressions::

        x*y
    """
    __slots__ = ()
    PRECEDENCE = 4

    def _compute_polynomial_degree(self, result):
        a, b = result
        if a == 0 and value(self._args_[0], exception=False) == 0:
            return 0
        if b == 0 and value(self._args_[1], exception=False) == 0:
            return 0
        if a is None or b is None:
            return None
        else:
            return a + b

    def getname(self, *args, **kwds):
        return 'prod'

    def _is_fixed(self, args):
        if all(args):
            return True
        for i in (0, 1):
            if args[i] and value(self._args_[i], exception=False) == 0:
                return True
        return False

    def _apply_operation(self, result):
        _l, _r = result
        return _l * _r

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f'{self.getname()}({', '.join(values)})'
        if values[0] in self._to_string.one:
            return values[1]
        if values[0] in self._to_string.minus_one:
            return f'- {values[1]}'
        return f'{values[0]}*{values[1]}'
    _to_string.one = {'1', '1.0', '(1)', '(1.0)'}
    _to_string.minus_one = {'-1', '-1.0', '(-1)', '(-1.0)'}