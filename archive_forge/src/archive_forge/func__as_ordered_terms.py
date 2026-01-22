from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef, UndefinedFunction, Function
def _as_ordered_terms(self, expr, order=None):
    """A compatibility function for ordering terms in Add. """
    order = order or self.order
    if order == 'old':
        return sorted(Add.make_args(expr), key=cmp_to_key(Basic._compare_pretty))
    elif order == 'none':
        return list(expr.args)
    else:
        return expr.as_ordered_terms(order=order)