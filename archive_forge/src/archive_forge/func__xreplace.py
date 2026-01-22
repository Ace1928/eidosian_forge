from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from itertools import chain, zip_longest
from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .core import ordering_of_classes
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name
from inspect import getmro
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
def _xreplace(self, rule):
    """
        Helper for xreplace. Tracks whether a replacement actually occurred.
        """
    if self in rule:
        return (rule[self], True)
    elif rule:
        args = []
        changed = False
        for a in self.args:
            _xreplace = getattr(a, '_xreplace', None)
            if _xreplace is not None:
                a_xr = _xreplace(rule)
                args.append(a_xr[0])
                changed |= a_xr[1]
            else:
                args.append(a)
        args = tuple(args)
        if changed:
            return (self.func(*args), True)
    return (self, False)