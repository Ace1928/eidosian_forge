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
class MultivariateTDistribution(JointDistribution):
    _argnames = ('mu', 'shape_mat', 'dof')
    is_Continuous = True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals ** k

    @staticmethod
    def check(mu, sigma, v):
        _value_check(mu.shape[0] == sigma.shape[0], 'Size of the location vector and shape matrix are incorrect.')
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite, 'The shape matrix must be positive definite. ')

    def pdf(self, *args):
        mu, sigma = (self.mu, self.shape_mat)
        v = S(self.dof)
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        x = args - mu
        return gamma((k + v) / 2) / (gamma(v / 2) * (v * pi) ** (k / 2) * sqrt(det(sigma))) * (1 + 1 / v * (x.transpose() * sigma_inv * x)[0]) ** ((-v - k) / 2)