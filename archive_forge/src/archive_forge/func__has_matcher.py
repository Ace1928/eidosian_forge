from __future__ import annotations
from operator import attrgetter
from collections import defaultdict
from sympy.utilities.exceptions import sympy_deprecation_warning
from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
def _has_matcher(self):
    """Helper for .has() that checks for containment of
        subexpressions within an expr by using sets of args
        of similar nodes, e.g. x + 1 in x + y + 1 checks
        to see that {x, 1} & {x, y, 1} == {x, 1}
        """

    def _ncsplit(expr):
        cpart, ncpart = sift(expr.args, lambda arg: arg.is_commutative is True, binary=True)
        return (set(cpart), ncpart)
    c, nc = _ncsplit(self)
    cls = self.__class__

    def is_in(expr):
        if isinstance(expr, cls):
            if expr == self:
                return True
            _c, _nc = _ncsplit(expr)
            if c & _c == c:
                if not nc:
                    return True
                elif len(nc) <= len(_nc):
                    for i in range(len(_nc) - len(nc) + 1):
                        if _nc[i:i + len(nc)] == nc:
                            return True
        return False
    return is_in