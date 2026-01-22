from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
from sympy.utilities.misc import filldedent
def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
    from sympy.matrices.common import MatrixOperations
    from sympy.core.basic import Basic
    if isinstance(arg, (Basic, MatrixOperations)):
        return doprint(arg)
    elif iterable(arg):
        if isinstance(arg, list):
            left, right = ('[', ']')
        elif isinstance(arg, tuple):
            left, right = ('(', ',)')
        else:
            raise NotImplementedError('unhandled type: %s, %s' % (type(arg), arg))
        return left + ', '.join((_recursive_to_string(doprint, e) for e in arg)) + right
    elif isinstance(arg, str):
        return arg
    else:
        return doprint(arg)