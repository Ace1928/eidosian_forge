from collections import OrderedDict
from collections.abc import MutableSet
from typing import Any, Callable
from .basic import Basic
from .sorting import default_sort_key, ordered
from .sympify import _sympify, sympify, _sympy_converter, SympifyError
from sympy.core.kind import Kind
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def wrap_tuples(*args, **kw_args):
    newargs = []
    for arg in args:
        if isinstance(arg, tuple):
            newargs.append(Tuple(*arg))
        else:
            newargs.append(arg)
    return method(*newargs, **kw_args)