from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
class principal_branch(Function):
    """
    Represent a polar number reduced to its principal branch on a quotient
    of the Riemann surface of the logarithm.

    Explanation
    ===========

    This is a function of two arguments. The first argument is a polar
    number `z`, and the second one a positive real number or infinity, `p`.
    The result is ``z mod exp_polar(I*p)``.

    Examples
    ========

    >>> from sympy import exp_polar, principal_branch, oo, I, pi
    >>> from sympy.abc import z
    >>> principal_branch(z, oo)
    z
    >>> principal_branch(exp_polar(2*pi*I)*3, 2*pi)
    3*exp_polar(0)
    >>> principal_branch(exp_polar(2*pi*I)*3*z, 2*pi)
    3*principal_branch(z, 2*pi)

    Parameters
    ==========

    x : Expr
        A polar number.

    period : Expr
        Positive real number or infinity.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    periodic_argument
    """
    is_polar = True
    is_comparable = False

    @classmethod
    def eval(self, x, period):
        from sympy.functions.elementary.exponential import exp_polar
        if isinstance(x, polar_lift):
            return principal_branch(x.args[0], period)
        if period == oo:
            return x
        ub = periodic_argument(x, oo)
        barg = periodic_argument(x, period)
        if ub != barg and (not ub.has(periodic_argument)) and (not barg.has(periodic_argument)):
            pl = polar_lift(x)

            def mr(expr):
                if not isinstance(expr, Symbol):
                    return polar_lift(expr)
                return expr
            pl = pl.replace(polar_lift, mr)
            ub = periodic_argument(pl, oo)
            if not pl.has(polar_lift):
                if ub != barg:
                    res = exp_polar(I * (barg - ub)) * pl
                else:
                    res = pl
                if not res.is_polar and (not res.has(exp_polar)):
                    res *= exp_polar(0)
                return res
        if not x.free_symbols:
            c, m = (x, ())
        else:
            c, m = x.as_coeff_mul(*x.free_symbols)
        others = []
        for y in m:
            if y.is_positive:
                c *= y
            else:
                others += [y]
        m = tuple(others)
        arg = periodic_argument(c, period)
        if arg.has(periodic_argument):
            return None
        if arg.is_number and (unbranched_argument(c) != arg or (arg == 0 and m != () and (c != 1))):
            if arg == 0:
                return abs(c) * principal_branch(Mul(*m), period)
            return principal_branch(exp_polar(I * arg) * Mul(*m), period) * abs(c)
        if arg.is_number and ((abs(arg) < period / 2) == True or arg == period / 2) and (m == ()):
            return exp_polar(arg * I) * abs(c)

    def _eval_evalf(self, prec):
        z, period = self.args
        p = periodic_argument(z, period)._eval_evalf(prec)
        if abs(p) > pi or p == -pi:
            return self
        from sympy.functions.elementary.exponential import exp
        return (abs(z) * exp(I * p))._eval_evalf(prec)