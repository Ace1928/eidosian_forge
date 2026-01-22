from __future__ import annotations
from typing import Any
from functools import wraps
from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def _handle_assign_to(expr, assign_to):
    if assign_to is None:
        return sympify(expr)
    if isinstance(assign_to, (list, tuple)):
        if len(expr) != len(assign_to):
            raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
        return CodeBlock(*[_handle_assign_to(lhs, rhs) for lhs, rhs in zip(expr, assign_to)])
    if isinstance(assign_to, str):
        if expr.is_Matrix:
            assign_to = MatrixSymbol(assign_to, *expr.shape)
        else:
            assign_to = Symbol(assign_to)
    elif not isinstance(assign_to, Basic):
        raise TypeError('{} cannot assign to object of type {}'.format(type(self).__name__, type(assign_to)))
    return Assignment(assign_to, expr)