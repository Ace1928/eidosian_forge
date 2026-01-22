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
def as_ordered_terms(self, order=None, data=False):
    """
        Transform an expression to an ordered list of terms.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x

        >>> (sin(x)**2*cos(x) + sin(x)**2 + 1).as_ordered_terms()
        [sin(x)**2*cos(x), sin(x)**2, 1]

        """
    from .numbers import Number, NumberSymbol
    if order is None and self.is_Add:
        key = lambda x: not isinstance(x, (Number, NumberSymbol))
        add_args = sorted(Add.make_args(self), key=key)
        if len(add_args) == 2 and isinstance(add_args[0], (Number, NumberSymbol)) and isinstance(add_args[1], Mul):
            mul_args = sorted(Mul.make_args(add_args[1]), key=key)
            if len(mul_args) == 2 and isinstance(mul_args[0], Number) and add_args[0].is_positive and mul_args[0].is_negative:
                return add_args
    key, reverse = self._parse_order(order)
    terms, gens = self.as_terms()
    if not any((term.is_Order for term, _ in terms)):
        ordered = sorted(terms, key=key, reverse=reverse)
    else:
        _terms, _order = ([], [])
        for term, repr in terms:
            if not term.is_Order:
                _terms.append((term, repr))
            else:
                _order.append((term, repr))
        ordered = sorted(_terms, key=key, reverse=True) + sorted(_order, key=key, reverse=True)
    if data:
        return (ordered, gens)
    else:
        return [term for term, _ in ordered]