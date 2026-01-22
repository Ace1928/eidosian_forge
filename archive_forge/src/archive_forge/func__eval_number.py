from typing import Tuple as tTuple
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core import Add, S
from sympy.core.evalf import get_integer_part, PrecisionExhausted
from sympy.core.function import Function
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import Integer
from sympy.core.relational import Gt, Lt, Ge, Le, Relational, is_eq
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import im, re
from sympy.multipledispatch import dispatch
@classmethod
def _eval_number(cls, arg):
    if arg.is_Number:
        return arg.ceiling()
    elif any((isinstance(i, j) for i in (arg, -arg) for j in (floor, ceiling))):
        return arg
    if arg.is_NumberSymbol:
        return arg.approximation_interval(Integer)[1]