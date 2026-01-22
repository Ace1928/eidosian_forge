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
class polar_lift(Function):
    """
    Lift argument to the Riemann surface of the logarithm, using the
    standard branch.

    Examples
    ========

    >>> from sympy import Symbol, polar_lift, I
    >>> p = Symbol('p', polar=True)
    >>> x = Symbol('x')
    >>> polar_lift(4)
    4*exp_polar(0)
    >>> polar_lift(-4)
    4*exp_polar(I*pi)
    >>> polar_lift(-I)
    exp_polar(-I*pi/2)
    >>> polar_lift(I + 2)
    polar_lift(2 + I)

    >>> polar_lift(4*x)
    4*polar_lift(x)
    >>> polar_lift(4*p)
    4*p

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    periodic_argument
    """
    is_polar = True
    is_comparable = False

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.complexes import arg as argument
        if arg.is_number:
            ar = argument(arg)
            if ar in (0, pi / 2, -pi / 2, pi):
                from sympy.functions.elementary.exponential import exp_polar
                return exp_polar(I * ar) * abs(arg)
        if arg.is_Mul:
            args = arg.args
        else:
            args = [arg]
        included = []
        excluded = []
        positive = []
        for arg in args:
            if arg.is_polar:
                included += [arg]
            elif arg.is_positive:
                positive += [arg]
            else:
                excluded += [arg]
        if len(excluded) < len(args):
            if excluded:
                return Mul(*included + positive) * polar_lift(Mul(*excluded))
            elif included:
                return Mul(*included + positive)
            else:
                from sympy.functions.elementary.exponential import exp_polar
                return Mul(*positive) * exp_polar(0)

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        return self.args[0]._eval_evalf(prec)

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)