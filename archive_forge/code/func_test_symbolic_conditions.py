from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
from sympy.stats.rv import Density
from sympy.testing.pytest import raises
def test_symbolic_conditions():
    B = Bernoulli('B', Rational(1, 4))
    D = Die('D', 4)
    b, n = symbols('b, n')
    Y = P(Eq(B, b))
    Z = E(D > n)
    assert Y == Piecewise((Rational(1, 4), Eq(b, 1)), (0, True)) + Piecewise((Rational(3, 4), Eq(b, 0)), (0, True))
    assert Z == Piecewise((Rational(1, 4), n < 1), (0, True)) + Piecewise((S.Half, n < 2), (0, True)) + Piecewise((Rational(3, 4), n < 3), (0, True)) + Piecewise((S.One, n < 4), (0, True))