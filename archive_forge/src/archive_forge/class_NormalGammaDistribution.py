from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import (Matrix, ones)
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import ImmutableMatrix, MatrixSymbol
from sympy.matrices.expressions.determinant import det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats.joint_rv import JointDistribution, JointPSpace, MarginalDistribution
from sympy.stats.rv import _value_check, random_symbols
class NormalGammaDistribution(JointDistribution):
    _argnames = ('mu', 'lamda', 'alpha', 'beta')
    is_Continuous = True

    @staticmethod
    def check(mu, lamda, alpha, beta):
        _value_check(mu.is_real, 'Location must be real.')
        _value_check(lamda > 0, 'Lambda must be positive')
        _value_check(alpha > 0, 'alpha must be positive')
        _value_check(beta > 0, 'beta must be positive')

    @property
    def set(self):
        return S.Reals * Interval(0, S.Infinity)

    def pdf(self, x, tau):
        beta, alpha, lamda = (self.beta, self.alpha, self.lamda)
        mu = self.mu
        return beta ** alpha * sqrt(lamda) / (gamma(alpha) * sqrt(2 * pi)) * tau ** (alpha - S.Half) * exp(-1 * beta * tau) * exp(-1 * (lamda * tau * (x - mu) ** 2) / S(2))

    def _marginal_distribution(self, indices, *sym):
        if len(indices) == 2:
            return self.pdf(*sym)
        if indices[0] == 0:
            x = sym[0]
            v, mu, sigma = (self.alpha - S.Half, self.mu, S(self.beta) / (self.lamda * self.alpha))
            return Lambda(sym, gamma((v + 1) / 2) / (gamma(v / 2) * sqrt(pi * v) * sigma) * (1 + 1 / v * ((x - mu) / sigma) ** 2) ** ((-v - 1) / 2))
        from sympy.stats.crv_types import GammaDistribution
        return Lambda(sym, GammaDistribution(self.alpha, self.beta)(sym[0]))