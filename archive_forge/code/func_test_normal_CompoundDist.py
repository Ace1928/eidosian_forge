from sympy.concrete.summations import Sum
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.sets import Interval
from sympy.stats import (Normal, P, E, density, Gamma, Poisson, Rayleigh,
from sympy.stats.compound_rv import CompoundDistribution, CompoundPSpace
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.frv_types import BernoulliDistribution
from sympy.testing.pytest import raises, ignore_warnings
from sympy.stats.joint_rv_types import MultivariateNormalDistribution
from sympy.abc import x
def test_normal_CompoundDist():
    X = Normal('X', 1, 2)
    Y = Normal('X', X, 4)
    assert density(Y)(x).simplify() == sqrt(10) * exp(-x ** 2 / 40 + x / 20 - S(1) / 40) / (20 * sqrt(pi))
    assert E(Y) == 1
    assert P(Y > 1) == S(1) / 2
    assert P(Y > 5).simplify() == S(1) / 2 - erf(sqrt(10) / 5) / 2
    assert variance(Y) == variance(X) + 4 ** 2