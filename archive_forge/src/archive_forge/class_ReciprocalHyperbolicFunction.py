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
class ReciprocalHyperbolicFunction(HyperbolicFunction):
    """Base class for reciprocal functions of hyperbolic functions. """
    _reciprocal_of = None
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)
        t = cls._reciprocal_of.eval(arg)
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]
        return 1 / t if t is not None else t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1 / t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1 / t

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_exp', arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_tractable', arg)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_tanh', arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_coth', arg)

    def as_real_imag(self, deep=True, **hints):
        return (1 / self._reciprocal_of(self.args[0])).as_real_imag(deep, **hints)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=True, **hints)
        return re_part + I * im_part

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal('_eval_expand_trig', **hints)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        return (1 / self._reciprocal_of(self.args[0]))._eval_as_leading_term(x)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0]).is_extended_real

    def _eval_is_finite(self):
        return (1 / self._reciprocal_of(self.args[0])).is_finite