from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime
from sympy.multipledispatch import MDNotImplementedError
from ..predicates.ntheory import (PrimePredicate, CompositePredicate,
def _PrimePredicate_number(expr, assumptions):
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    if exact:
        return isprime(i)