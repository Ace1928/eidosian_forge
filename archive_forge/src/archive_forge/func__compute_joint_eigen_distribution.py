from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.core.sympify import _sympify
from sympy.stats.rv import _symbol_converter, Density, RandomMatrixSymbol, is_random
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.random_matrix import RandomMatrixPSpace
from sympy.tensor.array import ArrayComprehension
def _compute_joint_eigen_distribution(self, beta):
    """
        Helper function to compute the joint distribution of phases
        of the complex eigen values of matrices belonging to any
        circular ensembles.
        """
    n = self.dimension
    Zbn = (2 * pi) ** n * (gamma(beta * n / 2 + 1) / S(gamma(beta / 2 + 1)) ** n)
    t = IndexedBase('t')
    i, j, k = (Dummy('i', integer=True), Dummy('j', integer=True), Dummy('k', integer=True))
    syms = ArrayComprehension(t[i], (i, 1, n)).doit()
    f = Product(Product(Abs(exp(I * t[k]) - exp(I * t[j])) ** beta, (j, k + 1, n)).doit(), (k, 1, n - 1)).doit()
    return Lambda(tuple(syms), f / Zbn)