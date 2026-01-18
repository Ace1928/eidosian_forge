from __future__ import annotations
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.utilities.misc import as_int
def untillist(cf):
    for nxt in cf:
        if isinstance(nxt, list):
            period.extend(nxt)
            yield x
            break
        yield nxt