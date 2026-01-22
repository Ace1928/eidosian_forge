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
@cacheit
def as_leading_term(self, *symbols, logx=None, cdir=0):
    """
        Returns the leading (nonzero) term of the series expansion of self.

        The _eval_as_leading_term routines are used to do this, and they must
        always return a non-zero value.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1 + x + x**2).as_leading_term(x)
        1
        >>> (1/x**2 + x + x**2).as_leading_term(x)
        x**(-2)

        """
    if len(symbols) > 1:
        c = self
        for x in symbols:
            c = c.as_leading_term(x, logx=logx, cdir=cdir)
        return c
    elif not symbols:
        return self
    x = sympify(symbols[0])
    if not x.is_symbol:
        raise ValueError('expecting a Symbol but got %s' % x)
    if x not in self.free_symbols:
        return self
    obj = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
    if obj is not None:
        from sympy.simplify.powsimp import powsimp
        return powsimp(obj, deep=True, combine='exp')
    raise NotImplementedError('as_leading_term(%s, %s)' % (self, x))