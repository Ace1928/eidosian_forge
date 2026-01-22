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
class cosh(HyperbolicFunction):
    """
    ``cosh(x)`` is the hyperbolic cosine of ``x``.

    The hyperbolic cosine function is $\\frac{e^x + e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import cosh
    >>> from sympy.abc import x
    >>> cosh(x)
    cosh(x)

    See Also
    ========

    sinh, tanh, acosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return sinh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.trigonometric import cos
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return S.One
            elif arg.is_negative:
                return cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN
            i_coeff = _imaginary_unit_as_coefficient(arg)
            if i_coeff is not None:
                return cos(i_coeff)
            elif arg.could_extract_minus_sign():
                return cls(-arg)
            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m * pi * I
                    return cosh(m) * cosh(x) + sinh(m) * sinh(x)
            if arg.is_zero:
                return S.One
            if arg.func == asinh:
                return sqrt(1 + arg.args[0] ** 2)
            if arg.func == acosh:
                return arg.args[0]
            if arg.func == atanh:
                return 1 / sqrt(1 - arg.args[0] ** 2)
            if arg.func == acoth:
                x = arg.args[0]
                return x / (sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x ** 2 / (n * (n - 1))
            else:
                return x ** n / factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
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
        return (cosh(re) * cos(im), sinh(re) * sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part * I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add:
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and (terms is not S.One):
                x = terms
                y = (coeff - 1) * x
        if x is not None:
            return (cosh(x) * cosh(y) + sinh(x) * sinh(y)).expand(trig=True)
        return cosh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(I * arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1 / sec(I * arg)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I * sinh(arg + pi * I / 2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half * arg) ** 2
        return (1 + tanh_half) / (1 - tanh_half)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half * arg) ** 2
        return (coth_half + 1) / (coth_half - 1)

    def _eval_rewrite_as_sech(self, arg, **kwargs):
        return 1 / sech(arg)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return S.One
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real or arg.is_imaginary:
            return True
        re, im = arg.as_real_imag()
        return (im % pi).is_zero

    def _eval_is_positive(self):
        z = self.args[0]
        x, y = z.as_real_imag()
        ymod = y % (2 * pi)
        yzero = ymod.is_zero
        if yzero:
            return True
        xzero = x.is_zero
        if xzero is False:
            return yzero
        return fuzzy_or([yzero, fuzzy_and([xzero, fuzzy_or([ymod < pi / 2, ymod > 3 * pi / 2])])])

    def _eval_is_nonnegative(self):
        z = self.args[0]
        x, y = z.as_real_imag()
        ymod = y % (2 * pi)
        yzero = ymod.is_zero
        if yzero:
            return True
        xzero = x.is_zero
        if xzero is False:
            return yzero
        return fuzzy_or([yzero, fuzzy_and([xzero, fuzzy_or([ymod <= pi / 2, ymod >= 3 * pi / 2])])])

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if ipi_mult and rest.is_zero:
            return (ipi_mult - S.Half).is_integer