from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random
class ChiDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, 'Number of degrees of freedom (k) must be positive.')
        _value_check(k.is_integer, 'Number of degrees of freedom (k) must be an integer.')
    set = Interval(0, oo)

    def pdf(self, x):
        return 2 ** (1 - self.k / 2) * x ** (self.k - 1) * exp(-x ** 2 / 2) / gamma(self.k / 2)

    def _characteristic_function(self, t):
        k = self.k
        part_1 = hyper((k / 2,), (S.Half,), -t ** 2 / 2)
        part_2 = I * t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (Rational(3, 2),), -t ** 2 / 2)
        return part_1 + part_2 * part_3

    def _moment_generating_function(self, t):
        k = self.k
        part_1 = hyper((k / 2,), (S.Half,), t ** 2 / 2)
        part_2 = t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (S(3) / 2,), t ** 2 / 2)
        return part_1 + part_2 * part_3