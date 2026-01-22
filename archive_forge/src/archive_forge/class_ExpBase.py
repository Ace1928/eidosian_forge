from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
class ExpBase(Function):
    unbranched = True
    _singularities = (S.ComplexInfinity,)

    @property
    def kind(self):
        return self.exp.kind

    def inverse(self, argindex=1):
        """
        Returns the inverse function of ``exp(x)``.
        """
        return log

    def as_numer_denom(self):
        """
        Returns this with a positive exponent as a 2-tuple (a fraction).

        Examples
        ========

        >>> from sympy import exp
        >>> from sympy.abc import x
        >>> exp(-x).as_numer_denom()
        (1, exp(x))
        >>> exp(x).as_numer_denom()
        (exp(x), 1)
        """
        exp = self.exp
        neg_exp = exp.is_negative
        if not neg_exp and (not (-exp).is_negative):
            neg_exp = exp.could_extract_minus_sign()
        if neg_exp:
            return (S.One, self.func(-exp))
        return (self, S.One)

    @property
    def exp(self):
        """
        Returns the exponent of the function.
        """
        return self.args[0]

    def as_base_exp(self):
        """
        Returns the 2-tuple (base, exponent).
        """
        return (self.func(1), Mul(*self.args))

    def _eval_adjoint(self):
        return self.func(self.exp.adjoint())

    def _eval_conjugate(self):
        return self.func(self.exp.conjugate())

    def _eval_transpose(self):
        return self.func(self.exp.transpose())

    def _eval_is_finite(self):
        arg = self.exp
        if arg.is_infinite:
            if arg.is_extended_negative:
                return True
            if arg.is_extended_positive:
                return False
        if arg.is_finite:
            return True

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            z = s.exp.is_zero
            if z:
                return True
            elif s.exp.is_rational and fuzzy_not(z):
                return False
        else:
            return s.is_rational

    def _eval_is_zero(self):
        return self.exp is S.NegativeInfinity

    def _eval_power(self, other):
        """exp(arg)**e -> exp(arg*e) if assumptions allow it.
        """
        b, e = self.as_base_exp()
        return Pow._eval_power(Pow(b, e, evaluate=False), other)

    def _eval_expand_power_exp(self, **hints):
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        arg = self.args[0]
        if arg.is_Add and arg.is_commutative:
            return Mul.fromiter((self.func(x) for x in arg.args))
        elif isinstance(arg, Sum) and arg.is_commutative:
            return Product(self.func(arg.function), *arg.limits)
        return self.func(arg)