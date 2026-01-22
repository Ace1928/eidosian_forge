from sympy.core import Basic, Add, sympify
from sympy.core.exprtools import gcd_terms
from sympy.utilities import public
from sympy.utilities.iterables import iterable
def _together(expr):
    if isinstance(expr, Basic):
        if expr.is_Atom or (expr.is_Function and (not deep)):
            return expr
        elif expr.is_Add:
            return gcd_terms(list(map(_together, Add.make_args(expr))), fraction=fraction)
        elif expr.is_Pow:
            base = _together(expr.base)
            if deep:
                exp = _together(expr.exp)
            else:
                exp = expr.exp
            return expr.__class__(base, exp)
        else:
            return expr.__class__(*[_together(arg) for arg in expr.args])
    elif iterable(expr):
        return expr.__class__([_together(ex) for ex in expr])
    return expr