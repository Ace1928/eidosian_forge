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
class coth(HyperbolicFunction):
    """
    ``coth(x)`` is the hyperbolic cotangent of ``x``.

    The hyperbolic cotangent function is $\\frac{\\cosh(x)}{\\sinh(x)}$.

    Examples
    ========

    >>> from sympy import coth
    >>> from sympy.abc import x
    >>> coth(x)
    coth(x)

    See Also
    ========

    sinh, cosh, acoth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / sinh(self.args[0]) ** 2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acoth

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN
            i_coeff = _imaginary_unit_as_coefficient(arg)
            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return I * cot(-i_coeff)
                return -I * cot(i_coeff)
            elif arg.could_extract_minus_sign():
                return -cls(-arg)
            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    cothm = coth(m * pi * I)
                    if cothm is S.ComplexInfinity:
                        return coth(x)
                    else:
                        return tanh(x)
            if arg.is_zero:
                return S.ComplexInfinity
            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1 + x ** 2) / x
            if arg.func == acosh:
                x = arg.args[0]
                return x / (sqrt(x - 1) * sqrt(x + 1))
            if arg.func == atanh:
                return 1 / arg.args[0]
            if arg.func == acoth:
                return arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            B = bernoulli(n + 1)
            F = factorial(n + 1)
            return 2 ** (n + 1) * B / F * x ** n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.trigonometric import cos, sin
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re) ** 2 + sin(im) ** 2
        return (sinh(re) * cosh(re) / denom, -sin(im) * cos(im) / denom)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = (exp(-arg), exp(arg))
        return (pos_exp + neg_exp) / (pos_exp - neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = (exp(-arg), exp(arg))
        return (pos_exp + neg_exp) / (pos_exp - neg_exp)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I * sinh(pi * I / 2 - arg) / sinh(arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I * cosh(arg) / cosh(pi * I / 2 - arg)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return 1 / tanh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)
        if x in arg.free_symbols and Order(1, x).contains(arg):
            return 1 / arg
        else:
            return self.func(arg)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            CX = [coth(x, evaluate=False)._eval_expand_trig() for x in arg.args]
            p = [[], []]
            n = len(arg.args)
            for i in range(n, -1, -1):
                p[(n - i) % 2].append(symmetric_poly(i, CX))
            return Add(*p[0]) / Add(*p[1])
        elif arg.is_Mul:
            coeff, x = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                c = coth(x, evaluate=False)
                p = [[], []]
                for i in range(coeff, -1, -1):
                    p[(coeff - i) % 2].append(binomial(coeff, i) * c ** i)
                return Add(*p[0]) / Add(*p[1])
        return coth(arg)