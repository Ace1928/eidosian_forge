from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int
def _parse_symbols(symbols):
    if not symbols:
        return ()
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True)
    elif isinstance(symbols, (Expr, FreeGroupElement)):
        return (symbols,)
    elif is_sequence(symbols):
        if all((isinstance(s, str) for s in symbols)):
            return _symbols(symbols)
        elif all((isinstance(s, Expr) for s in symbols)):
            return symbols
    raise ValueError('The type of `symbols` must be one of the following: a str, Symbol/Expr or a sequence of one of these types')