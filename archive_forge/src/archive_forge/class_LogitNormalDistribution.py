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
class LogitNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')
    set = Interval.open(0, 1)

    @staticmethod
    def check(mu, s):
        _value_check((s ** 2).is_real is not False and s ** 2 > 0, 'Squared scale parameter s must be positive.')
        _value_check(mu.is_real is not False, 'Location parameter must be real')

    def _logit(self, x):
        return log(x / (1 - x))

    def pdf(self, x):
        mu, s = (self.mu, self.s)
        return exp(-(self._logit(x) - mu) ** 2 / (2 * s ** 2)) * (S.One / sqrt(2 * pi * s ** 2)) * (1 / (x * (1 - x)))

    def _cdf(self, x):
        mu, s = (self.mu, self.s)
        return S.One / 2 * (1 + erf((self._logit(x) - mu) / sqrt(2 * s ** 2)))