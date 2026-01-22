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
class MultivariateEwensDistribution(JointDistribution):
    _argnames = ('n', 'theta')
    is_Discrete = True
    is_Continuous = False

    @staticmethod
    def check(n, theta):
        _value_check(n > 0, 'sample size should be positive integer.')
        _value_check(theta.is_positive, 'mutation rate should be positive.')

    @property
    def set(self):
        if not isinstance(self.n, Integer):
            i = Symbol('i', integer=True, positive=True)
            return Product(Intersection(S.Naturals0, Interval(0, self.n // i)), (i, 1, self.n))
        prod_set = Range(0, self.n + 1)
        for i in range(2, self.n + 1):
            prod_set *= Range(0, self.n // i + 1)
        return prod_set.flatten()

    def pdf(self, *syms):
        n, theta = (self.n, self.theta)
        condi = isinstance(self.n, Integer)
        if not (isinstance(syms[0], IndexedBase) or condi):
            raise ValueError('Please use IndexedBase object for syms as the dimension is symbolic')
        term_1 = factorial(n) / rf(theta, n)
        if condi:
            term_2 = Mul.fromiter((theta ** syms[j] / ((j + 1) ** syms[j] * factorial(syms[j])) for j in range(n)))
            cond = Eq(sum([(k + 1) * syms[k] for k in range(n)]), n)
            return Piecewise((term_1 * term_2, cond), (0, True))
        syms = syms[0]
        j, k = symbols('j, k', positive=True, integer=True)
        term_2 = Product(theta ** syms[j] / ((j + 1) ** syms[j] * factorial(syms[j])), (j, 0, n - 1))
        cond = Eq(Sum((k + 1) * syms[k], (k, 0, n - 1)), n)
        return Piecewise((term_1 * term_2, cond), (0, True))