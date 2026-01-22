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
class LogLogisticDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, 'Scale parameter Alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter Beta must be positive.')

    def pdf(self, x):
        a, b = (self.alpha, self.beta)
        return b / a * (x / a) ** (b - 1) / (1 + (x / a) ** b) ** 2

    def _cdf(self, x):
        a, b = (self.alpha, self.beta)
        return 1 / (1 + (x / a) ** (-b))

    def _quantile(self, p):
        a, b = (self.alpha, self.beta)
        return a * (p / (1 - p)) ** (1 / b)

    def expectation(self, expr, var, **kwargs):
        a, b = self.args
        return Piecewise((S.NaN, b <= 1), (pi * a / (b * sin(pi / b)), True))