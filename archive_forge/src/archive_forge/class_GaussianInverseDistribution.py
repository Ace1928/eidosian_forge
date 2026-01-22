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
class GaussianInverseDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'shape')

    @property
    def set(self):
        return Interval(0, oo)

    @staticmethod
    def check(mean, shape):
        _value_check(shape > 0, 'Shape parameter must be positive')
        _value_check(mean > 0, 'Mean must be positive')

    def pdf(self, x):
        mu, s = (self.mean, self.shape)
        return exp(-s * (x - mu) ** 2 / (2 * x * mu ** 2)) * sqrt(s / (2 * pi * x ** 3))

    def _cdf(self, x):
        from sympy.stats import cdf
        mu, s = (self.mean, self.shape)
        stdNormalcdf = cdf(Normal('x', 0, 1))
        first_term = stdNormalcdf(sqrt(s / x) * (x / mu - S.One))
        second_term = exp(2 * s / mu) * stdNormalcdf(-sqrt(s / x) * (x / mu + S.One))
        return first_term + second_term

    def _characteristic_function(self, t):
        mu, s = (self.mean, self.shape)
        return exp(s / mu * (1 - sqrt(1 - 2 * mu ** 2 * I * t / s)))

    def _moment_generating_function(self, t):
        mu, s = (self.mean, self.shape)
        return exp(s / mu * (1 - sqrt(1 - 2 * mu ** 2 * t / s)))