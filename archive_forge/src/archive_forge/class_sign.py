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
class sign(Function):
    """
    Returns the complex sign of an expression:

    Explanation
    ===========

    If the expression is real the sign will be:

        * $1$ if expression is positive
        * $0$ if expression is equal to zero
        * $-1$ if expression is negative

    If the expression is imaginary the sign will be:

        * $I$ if im(expression) is positive
        * $-I$ if im(expression) is negative

    Otherwise an unevaluated expression will be returned. When evaluated, the
    result (in general) will be ``cos(arg(expr)) + I*sin(arg(expr))``.

    Examples
    ========

    >>> from sympy import sign, I

    >>> sign(-1)
    -1
    >>> sign(0)
    0
    >>> sign(-3*I)
    -I
    >>> sign(1 + I)
    sign(1 + I)
    >>> _.evalf()
    0.707106781186548 + 0.707106781186548*I

    Parameters
    ==========

    arg : Expr
        Real or imaginary expression.

    Returns
    =======

    expr : Expr
        Complex sign of expression.

    See Also
    ========

    Abs, conjugate
    """
    is_complex = True
    _singularities = True

    def doit(self, **hints):
        s = super().doit()
        if s == self and self.args[0].is_zero is False:
            return self.args[0] / Abs(self.args[0])
        return s

    @classmethod
    def eval(cls, arg):
        if arg.is_Mul:
            c, args = arg.as_coeff_mul()
            unk = []
            s = sign(c)
            for a in args:
                if a.is_extended_negative:
                    s = -s
                elif a.is_extended_positive:
                    pass
                elif a.is_imaginary:
                    ai = im(a)
                    if ai.is_comparable:
                        s *= I
                        if ai.is_extended_negative:
                            s = -s
                    else:
                        unk.append(a)
                else:
                    unk.append(a)
            if c is S.One and len(unk) == len(args):
                return None
            return s * cls(arg._new_rawargs(*unk))
        if arg is S.NaN:
            return S.NaN
        if arg.is_zero:
            return S.Zero
        if arg.is_extended_positive:
            return S.One
        if arg.is_extended_negative:
            return S.NegativeOne
        if arg.is_Function:
            if isinstance(arg, sign):
                return arg
        if arg.is_imaginary:
            if arg.is_Pow and arg.exp is S.Half:
                return I
            arg2 = -I * arg
            if arg2.is_extended_positive:
                return I
            if arg2.is_extended_negative:
                return -I

    def _eval_Abs(self):
        if fuzzy_not(self.args[0].is_zero):
            return S.One

    def _eval_conjugate(self):
        return sign(conjugate(self.args[0]))

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) * DiracDelta(self.args[0])
        elif self.args[0].is_imaginary:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) * DiracDelta(-I * self.args[0])

    def _eval_is_nonnegative(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_nonpositive(self):
        if self.args[0].is_nonpositive:
            return True

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_power(self, other):
        if fuzzy_not(self.args[0].is_zero) and other.is_integer and other.is_even:
            return S.One

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg0 = self.args[0]
        x0 = arg0.subs(x, 0)
        if x0 != 0:
            return self.func(x0)
        if cdir != 0:
            cdir = arg0.dir(x, cdir)
        return -S.One if re(cdir) < 0 else S.One

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((1, arg > 0), (-1, arg < 0), (0, True))

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return Heaviside(arg) * 2 - 1

    def _eval_rewrite_as_Abs(self, arg, **kwargs):
        return Piecewise((0, Eq(arg, 0)), (arg / Abs(arg), True))

    def _eval_simplify(self, **kwargs):
        return self.func(factor_terms(self.args[0]))