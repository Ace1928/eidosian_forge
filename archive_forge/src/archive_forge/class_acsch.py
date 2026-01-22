from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.polys.specialpolys import symmetric_poly
class acsch(InverseHyperbolicFunction):
    """
    ``acsch(x)`` is the inverse hyperbolic cosecant of ``x``.

    The inverse hyperbolic cosecant function.

    Examples
    ========

    >>> from sympy import acsch, sqrt, I
    >>> from sympy.abc import x
    >>> acsch(x).diff(x)
    -1/(x**2*sqrt(1 + x**(-2)))
    >>> acsch(1).diff(x)
    0
    >>> acsch(1)
    log(1 + sqrt(2))
    >>> acsch(I)
    -I*pi/2
    >>> acsch(-2*I)
    I*pi/6
    >>> acsch(I*(sqrt(6) - sqrt(2)))
    -5*I*pi/12

    See Also
    ========

    asinh

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsch/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1 / (z ** 2 * sqrt(1 + 1 / z ** 2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return log(1 + sqrt(2))
            elif arg is S.NegativeOne:
                return -log(1 + sqrt(2))
        if arg.is_number:
            cst_table = _acsch_table()
            if arg in cst_table:
                return cst_table[arg] * I
        if arg is S.ComplexInfinity:
            return S.Zero
        if arg.is_infinite:
            return S.Zero
        if arg.is_zero:
            return S.ComplexInfinity
        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return -p * ((n - 1) * (n - 2)) * x ** 2 / (4 * (n // 2) ** 2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) * n
                F = factorial(k) * n // 2 * n // 2
                return S.NegativeOne ** (k + 1) * R / F * x ** n / 4

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 in (-I, I, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if x0 is S.ComplexInfinity:
            return (1 / arg).as_leading_term(x)
        if x0.is_imaginary and (1 + x0 ** 2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return -self.func(x0) - I * pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return -self.func(x0) + I * pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        if arg0 is I:
            t = Dummy('t', positive=True)
            ser = acsch(I + t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = -I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else -I * pi / 2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            res = ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
            return res
        if arg0 == S.NegativeOne * I:
            t = Dummy('t', positive=True)
            ser = acsch(-I + t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else I * pi / 2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        if arg0.is_imaginary and (1 + arg0 ** 2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return -res - I * pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return -res + I * pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csch

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1 / arg + sqrt(1 / arg ** 2 + 1))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return asinh(1 / arg)

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return I * (sqrt(1 - I / arg) / sqrt(I / arg - 1) * acosh(I / arg) - pi * S.Half)

    def _eval_rewrite_as_atanh(self, arg, **kwargs):
        arg2 = arg ** 2
        arg2p1 = arg2 + 1
        return sqrt(-arg2) / arg * (pi * S.Half - sqrt(-arg2p1 ** 2) / arg2p1 * atanh(sqrt(arg2p1)))

    def _eval_is_zero(self):
        return self.args[0].is_infinite