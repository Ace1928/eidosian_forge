from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
def theano_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Theano graph variable.

    .. deprecated:: 1.8

      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    Returns
    =======

    theano.gof.graph.Variable
        A variable corresponding to the expression's value in a Theano symbolic
        expression graph.

    """
    sympy_deprecation_warning('\n        sympy.printing.theanocode is deprecated. Theano has been renamed to\n        Aesara. Use sympy.printing.aesaracode instead.', deprecated_since_version='1.8', active_deprecations_target='theanocode-deprecated')
    if not theano:
        raise ImportError('theano is required for theano_code')
    if cache is None:
        cache = global_cache
    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)