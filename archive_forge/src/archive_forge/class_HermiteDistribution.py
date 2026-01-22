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
class HermiteDistribution(SingleDiscreteDistribution):
    _argnames = ('a1', 'a2')
    set = S.Naturals0

    @staticmethod
    def check(a1, a2):
        _value_check(a1.is_nonnegative, 'Parameter a1 must be >= 0.')
        _value_check(a2.is_nonnegative, 'Parameter a2 must be >= 0.')

    def pdf(self, k):
        a1, a2 = (self.a1, self.a2)
        term1 = exp(-(a1 + a2))
        j = Dummy('j', integer=True)
        num = a1 ** (k - 2 * j) * a2 ** j
        den = factorial(k - 2 * j) * factorial(j)
        return term1 * Sum(num / den, (j, 0, k // 2)).doit()

    def _moment_generating_function(self, t):
        a1, a2 = (self.a1, self.a2)
        term1 = a1 * (exp(t) - 1)
        term2 = a2 * (exp(2 * t) - 1)
        return exp(term1 + term2)

    def _characteristic_function(self, t):
        a1, a2 = (self.a1, self.a2)
        term1 = a1 * (exp(I * t) - 1)
        term2 = a2 * (exp(2 * I * t) - 1)
        return exp(term1 + term2)