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
class asinh(InverseHyperbolicFunction):
    """
    ``asinh(x)`` is the inverse hyperbolic sine of ``x``.

    The inverse hyperbolic sine function.

    Examples
    ========

    >>> from sympy import asinh
    >>> from sympy.abc import x
    >>> asinh(x).diff(x)
    1/sqrt(x**2 + 1)
    >>> asinh(1)
    log(1 + sqrt(2))

    See Also
    ========

    acosh, atanh, sinh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / sqrt(self.args[0] ** 2 + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return log(sqrt(2) + 1)
            elif arg is S.NegativeOne:
                return log(sqrt(2) - 1)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.ComplexInfinity
            if arg.is_zero:
                return S.Zero
            i_coeff = _imaginary_unit_as_coefficient(arg)
            if i_coeff is not None:
                return I * asin(i_coeff)
            elif arg.could_extract_minus_sign():
                return -cls(-arg)
        if isinstance(arg, sinh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor((i + pi / 2) / pi)
                m = z - I * pi * f
                even = f.is_even
                if even is True:
                    return m
                elif even is False:
                    return -m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return -p * (n - 2) ** 2 / (n * (n - 1)) * x ** 2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return S.NegativeOne ** k * R / F * x ** n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)
        if x0 in (-I, I, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if (1 + x0 ** 2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_negative:
                    return -self.func(x0) - I * pi
            elif re(ndir).is_negative:
                if im(x0).is_positive:
                    return -self.func(x0) + I * pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        if arg0 in (I, -I):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        if (1 + arg0 ** 2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_negative:
                    return -res - I * pi
            elif re(ndir).is_negative:
                if im(arg0).is_positive:
                    return -res + I * pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x ** 2 + 1))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(x / sqrt(1 + x ** 2))

    def _eval_rewrite_as_acosh(self, x, **kwargs):
        ix = I * x
        return I * (sqrt(1 - ix) / sqrt(ix - 1) * acosh(ix) - pi / 2)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return -I * asin(I * x)

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return I * acos(I * x) - I * pi / 2

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sinh

    def _eval_is_zero(self):
        return self.args[0].is_zero