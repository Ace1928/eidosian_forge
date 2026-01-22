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
@classmethod
def _parse_order(cls, order):
    """Parse and configure the ordering of terms. """
    from sympy.polys.orderings import monomial_key
    startswith = getattr(order, 'startswith', None)
    if startswith is None:
        reverse = False
    else:
        reverse = startswith('rev-')
        if reverse:
            order = order[4:]
    monom_key = monomial_key(order)

    def neg(monom):
        return tuple([neg(m) if isinstance(m, tuple) else -m for m in monom])

    def key(term):
        _, ((re, im), monom, ncpart) = term
        monom = neg(monom_key(monom))
        ncpart = tuple([e.sort_key(order=order) for e in ncpart])
        coeff = ((bool(im), im), (re, im))
        return (monom, ncpart, coeff)
    return (key, reverse)