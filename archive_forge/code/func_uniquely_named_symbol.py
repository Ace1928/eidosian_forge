from __future__ import annotations
from .assumptions import StdFactKB, _assume_defined
from .basic import Basic, Atom
from .cache import cacheit
from .containers import Tuple
from .expr import Expr, AtomicExpr
from .function import AppliedUndef, FunctionClass
from .kind import NumberKind, UndefinedKind
from .logic import fuzzy_bool
from .singleton import S
from .sorting import ordered
from .sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.misc import filldedent
import string
import re as _re
import random
from itertools import product
from typing import Any
def uniquely_named_symbol(xname, exprs=(), compare=str, modify=None, **assumptions):
    """
    Return a symbol whose name is derivated from *xname* but is unique
    from any other symbols in *exprs*.

    *xname* and symbol names in *exprs* are passed to *compare* to be
    converted to comparable forms. If ``compare(xname)`` is not unique,
    it is recursively passed to *modify* until unique name is acquired.

    Parameters
    ==========

    xname : str or Symbol
        Base name for the new symbol.

    exprs : Expr or iterable of Expr
        Expressions whose symbols are compared to *xname*.

    compare : function
        Unary function which transforms *xname* and symbol names from
        *exprs* to comparable form.

    modify : function
        Unary function which modifies the string. Default is appending
        the number, or increasing the number if exists.

    Examples
    ========

    By default, a number is appended to *xname* to generate unique name.
    If the number already exists, it is recursively increased.

    >>> from sympy.core.symbol import uniquely_named_symbol, Symbol
    >>> uniquely_named_symbol('x', Symbol('x'))
    x0
    >>> uniquely_named_symbol('x', (Symbol('x'), Symbol('x0')))
    x1
    >>> uniquely_named_symbol('x0', (Symbol('x1'), Symbol('x0')))
    x2

    Name generation can be controlled by passing *modify* parameter.

    >>> from sympy.abc import x
    >>> uniquely_named_symbol('x', x, modify=lambda s: 2*s)
    xx

    """

    def numbered_string_incr(s, start=0):
        if not s:
            return str(start)
        i = len(s) - 1
        while i != -1:
            if not s[i].isdigit():
                break
            i -= 1
        n = str(int(s[i + 1:] or start - 1) + 1)
        return s[:i + 1] + n
    default = None
    if is_sequence(xname):
        xname, default = xname
    x = compare(xname)
    if not exprs:
        return _symbol(x, default, **assumptions)
    if not is_sequence(exprs):
        exprs = [exprs]
    names = set().union([i.name for e in exprs for i in e.atoms(Symbol)] + [i.func.name for e in exprs for i in e.atoms(AppliedUndef)])
    if modify is None:
        modify = numbered_string_incr
    while any((x == compare(s) for s in names)):
        x = modify(x)
    return _symbol(x, default, **assumptions)