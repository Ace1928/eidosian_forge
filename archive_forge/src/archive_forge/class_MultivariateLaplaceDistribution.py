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
class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')
    is_Continuous = True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals ** k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0], 'Size of the mean vector and covariance matrix are incorrect.')
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite, 'The covariance matrix must be positive definite. ')

    def pdf(self, *args):
        mu, sigma = (self.mu, self.sigma)
        mu_T = mu.transpose()
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        args_T = args.transpose()
        x = (mu_T * sigma_inv * mu)[0]
        y = (args_T * sigma_inv * args)[0]
        v = 1 - k / 2
        return 2 * (y / (2 + x)) ** (v / 2) * besselk(v, sqrt((2 + x) * y)) * exp((args_T * sigma_inv * mu)[0]) / ((2 * pi) ** (k / 2) * sqrt(det(sigma)))