from __future__ import annotations
from typing import Any, Callable
from inspect import getmro
import string
from sympy.core.random import choice
from .parameters import global_parameters
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
from .basic import Basic
class SympifyError(ValueError):

    def __init__(self, expr, base_exc=None):
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if self.base_exc is None:
            return 'SympifyError: %r' % (self.expr,)
        return "Sympify of expression '%s' failed, because of exception being raised:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__, str(self.base_exc))