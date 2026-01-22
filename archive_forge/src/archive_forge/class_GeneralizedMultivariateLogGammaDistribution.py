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
class GeneralizedMultivariateLogGammaDistribution(JointDistribution):
    _argnames = ('delta', 'v', 'lamda', 'mu')
    is_Continuous = True

    def check(self, delta, v, l, mu):
        _value_check((delta >= 0, delta <= 1), 'delta must be in range [0, 1].')
        _value_check(v > 0, 'v must be positive')
        for lk in l:
            _value_check(lk > 0, 'lamda must be a positive vector.')
        for muk in mu:
            _value_check(muk > 0, 'mu must be a positive vector.')
        _value_check(len(l) > 1, 'the distribution should have at least two random variables.')

    @property
    def set(self):
        return S.Reals ** len(self.lamda)

    def pdf(self, *y):
        d, v, l, mu = (self.delta, self.v, self.lamda, self.mu)
        n = Symbol('n', negative=False, integer=True)
        k = len(l)
        sterm1 = Pow(1 - d, n) / (gamma(v + n) ** (k - 1) * gamma(v) * gamma(n + 1))
        sterm2 = Mul.fromiter((mui * li ** (-v - n) for mui, li in zip(mu, l)))
        term1 = sterm1 * sterm2
        sterm3 = (v + n) * sum([mui * yi for mui, yi in zip(mu, y)])
        sterm4 = sum([exp(mui * yi) / li for mui, yi, li in zip(mu, y, l)])
        term2 = exp(sterm3 - sterm4)
        return Pow(d, v) * Sum(term1 * term2, (n, 0, S.Infinity))