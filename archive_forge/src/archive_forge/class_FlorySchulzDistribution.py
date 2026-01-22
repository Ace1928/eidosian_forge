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
class FlorySchulzDistribution(SingleDiscreteDistribution):
    _argnames = ('a',)
    set = S.Naturals

    @staticmethod
    def check(a):
        _value_check((0 < a, a < 1), 'a must be between 0 and 1')

    def pdf(self, k):
        a = self.a
        return a ** 2 * k * (1 - a) ** (k - 1)

    def _characteristic_function(self, t):
        a = self.a
        return a ** 2 * exp(I * t) / (1 + (a - 1) * exp(I * t)) ** 2

    def _moment_generating_function(self, t):
        a = self.a
        return a ** 2 * exp(t) / (1 + (a - 1) * exp(t)) ** 2