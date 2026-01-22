import operator
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .numvalue import is_potentially_variable, is_constant
from .visitor import polynomial_degree
class RangedExpression(RelationalExpression):
    """
    Ranged expressions, which define relations with a lower and upper bound::

        x < y < z
        x <= y <= z

    args:
        args (tuple): child nodes
        strict (tuple): flags that indicate whether the inequalities are strict
    """
    __slots__ = ('_strict',)
    PRECEDENCE = 9
    STRICT = {False: (False, False), True: (True, True), (True, True): (True, True), (False, False): (False, False), (True, False): (True, False), (False, True): (False, True)}

    def __init__(self, args, strict):
        super(RangedExpression, self).__init__(args)
        self._strict = RangedExpression.STRICT[strict]

    def nargs(self):
        return 3

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def _apply_operation(self, result):
        _l, _b, _r = result
        if not self._strict[0]:
            if not self._strict[1]:
                return _l <= _b and _b <= _r
            else:
                return _l <= _b and _b < _r
        elif not self._strict[1]:
            return _l < _b and _b <= _r
        else:
            return _l < _b and _b < _r

    def _to_string(self, values, verbose, smap):
        return '%s  %s  %s  %s  %s' % (values[0], '<='[:2 - self._strict[0]], values[1], '<='[:2 - self._strict[1]], values[2])

    @property
    def strict(self):
        return self._strict