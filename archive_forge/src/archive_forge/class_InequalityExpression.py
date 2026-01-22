import operator
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .numvalue import is_potentially_variable, is_constant
from .visitor import polynomial_degree
class InequalityExpression(RelationalExpression):
    """
    Inequality expressions, which define less-than or
    less-than-or-equal relations::

        x < y
        x <= y

    args:
        args (tuple): child nodes
        strict (bool): a flag that indicates whether the inequality is strict
    """
    __slots__ = ('_strict',)
    PRECEDENCE = 9

    def __init__(self, args, strict):
        super().__init__(args)
        self._strict = strict

    def nargs(self):
        return 2

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def _apply_operation(self, result):
        _l, _r = result
        if self._strict:
            return _l < _r
        return _l <= _r

    def _to_string(self, values, verbose, smap):
        return '%s  %s  %s' % (values[0], '<='[:2 - self._strict], values[1])

    @property
    def strict(self):
        return self._strict