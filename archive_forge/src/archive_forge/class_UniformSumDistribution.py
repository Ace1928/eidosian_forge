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
class UniformSumDistribution(SingleContinuousDistribution):
    _argnames = ('n',)

    @property
    def set(self):
        return Interval(0, self.n)

    @staticmethod
    def check(n):
        _value_check((n > 0, n.is_integer), 'Parameter n must be positive integer.')

    def pdf(self, x):
        n = self.n
        k = Dummy('k')
        return 1 / factorial(n - 1) * Sum((-1) ** k * binomial(n, k) * (x - k) ** (n - 1), (k, 0, floor(x)))

    def _cdf(self, x):
        n = self.n
        k = Dummy('k')
        return Piecewise((S.Zero, x < 0), (1 / factorial(n) * Sum((-1) ** k * binomial(n, k) * (x - k) ** n, (k, 0, floor(x))), x <= n), (S.One, True))

    def _characteristic_function(self, t):
        return ((exp(I * t) - 1) / (I * t)) ** self.n

    def _moment_generating_function(self, t):
        return ((exp(t) - 1) / t) ** self.n