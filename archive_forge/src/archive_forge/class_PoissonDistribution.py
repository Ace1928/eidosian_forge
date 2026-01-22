from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import (polylog, zeta)
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random
class PoissonDistribution(SingleDiscreteDistribution):
    _argnames = ('lamda',)
    set = S.Naturals0

    @staticmethod
    def check(lamda):
        _value_check(lamda > 0, 'Lambda must be positive')

    def pdf(self, k):
        return self.lamda ** k / factorial(k) * exp(-self.lamda)

    def _characteristic_function(self, t):
        return exp(self.lamda * (exp(I * t) - 1))

    def _moment_generating_function(self, t):
        return exp(self.lamda * (exp(t) - 1))