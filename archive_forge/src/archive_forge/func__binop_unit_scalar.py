import functools
import operator
from matplotlib import _api
def _binop_unit_scalar(self, op, scalar):
    """Combine *self* and *scalar* using *op*."""
    return UnitDbl(op(self._value, scalar), self._units)