from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable
from functools import reduce
import re
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .sorting import default_sort_key
from .kind import NumberKind
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, sift
from mpmath.libmp import mpf_log, prec_to_dps
from mpmath.libmp.libintmath import giant_steps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal
@staticmethod
def _expand_hint(expr, hint, deep=True, **hints):
    """
        Helper for ``expand()``.  Recursively calls ``expr._eval_expand_hint()``.

        Returns ``(expr, hit)``, where expr is the (possibly) expanded
        ``expr`` and ``hit`` is ``True`` if ``expr`` was truly expanded and
        ``False`` otherwise.
        """
    hit = False
    if deep and getattr(expr, 'args', ()) and (not expr.is_Atom):
        sargs = []
        for arg in expr.args:
            arg, arghit = Expr._expand_hint(arg, hint, **hints)
            hit |= arghit
            sargs.append(arg)
        if hit:
            expr = expr.func(*sargs)
    if hasattr(expr, hint):
        newexpr = getattr(expr, hint)(**hints)
        if newexpr != expr:
            return (newexpr, True)
    return (expr, hit)