import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def _all_keys(self, attr):
    _keys = getattr(self, attr)
    _all = set() if _keys is None else set(_keys)
    if self.args is not None:
        for arg in self.args:
            if isinstance(arg, Expr):
                _all = _all.union(arg._all_keys(attr))
    return _all