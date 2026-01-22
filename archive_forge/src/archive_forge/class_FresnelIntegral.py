from sympy.core import EulerGamma  # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, expand_mul
from sympy.core.numbers import I, pi, Rational
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg
class FresnelIntegral(Function):
    """ Base class for the Fresnel integrals."""
    unbranched = True

    @classmethod
    def eval(cls, z):
        if z is S.Infinity:
            return S.Half
        if z.is_zero:
            return S.Zero
        prefact = S.One
        newarg = z
        changed = False
        nz = newarg.extract_multiplicatively(-1)
        if nz is not None:
            prefact = -prefact
            newarg = nz
            changed = True
        nz = newarg.extract_multiplicatively(I)
        if nz is not None:
            prefact = cls._sign * I * prefact
            newarg = nz
            changed = True
        if changed:
            return prefact * cls(newarg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self._trigfunc(S.Half * pi * self.args[0] ** 2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real
    _eval_is_finite = _eval_is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())
    as_real_imag = real_to_real_as_real_imag