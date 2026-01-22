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
class MoyalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.is_real, 'Location parameter must be real.')
        _value_check(sigma.is_real and sigma > 0, 'Scale parameter must be real        and positive.')

    def pdf(self, x):
        mu, sigma = (self.mu, self.sigma)
        num = exp(-(exp(-(x - mu) / sigma) + (x - mu) / sigma) / 2)
        den = sqrt(2 * pi) * sigma
        return num / den

    def _characteristic_function(self, t):
        mu, sigma = (self.mu, self.sigma)
        term1 = exp(I * t * mu)
        term2 = 2 ** (-I * sigma * t) * gamma(Rational(1, 2) - I * t * sigma)
        return term1 * term2 / sqrt(pi)

    def _moment_generating_function(self, t):
        mu, sigma = (self.mu, self.sigma)
        term1 = exp(t * mu)
        term2 = 2 ** (-1 * sigma * t) * gamma(Rational(1, 2) - t * sigma)
        return term1 * term2 / sqrt(pi)