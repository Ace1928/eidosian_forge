from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
def aesara_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Aesara graph variable.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    Returns
    =======

    aesara.graph.basic.Variable
        A variable corresponding to the expression's value in a Aesara symbolic
        expression graph.

    """
    if not aesara:
        raise ImportError('aesara is required for aesara_code')
    if cache is None:
        cache = global_cache
    return AesaraPrinter(cache=cache, settings={}).doprint(expr, **kwargs)