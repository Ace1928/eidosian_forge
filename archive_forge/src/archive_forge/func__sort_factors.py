from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _sort_factors(factors, **args):
    """Sort low-level factors in increasing 'complexity' order. """

    def order_if_multiple_key(factor):
        f, n = factor
        return (len(f), n, f)

    def order_no_multiple_key(f):
        return (len(f), f)
    if args.get('multiple', True):
        return sorted(factors, key=order_if_multiple_key)
    else:
        return sorted(factors, key=order_no_multiple_key)