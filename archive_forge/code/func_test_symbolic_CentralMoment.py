from sympy.concrete.summations import Sum
from sympy.core.mul import Mul
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.core.expr import unchanged
from sympy.stats import (Normal, Poisson, variance, Covariance, Variance,
from sympy.stats.rv import probability, expectation
def test_symbolic_CentralMoment():
    mu = symbols('mu', real=True)
    sigma = symbols('sigma', positive=True)
    x = symbols('x')
    X = Normal('X', mu, sigma)
    CM = CentralMoment(X, 6)
    assert CM.rewrite(Expectation) == Expectation((X - Expectation(X)) ** 6)
    assert CM.rewrite(Probability) == Integral((x - Integral(x * Probability(True), (x, -oo, oo))) ** 6 * Probability(Eq(X, x)), (x, -oo, oo))
    k = Dummy('k')
    expri = Integral(sqrt(2) * (k - Integral(sqrt(2) * k * exp(-(k - mu) ** 2 / (2 * sigma ** 2)) / (2 * sqrt(pi) * sigma), (k, -oo, oo))) ** 6 * exp(-(k - mu) ** 2 / (2 * sigma ** 2)) / (2 * sqrt(pi) * sigma), (k, -oo, oo))
    assert CM.rewrite(Integral).dummy_eq(expri)
    assert CM.doit().simplify() == 15 * sigma ** 6
    CM = Moment(5, 5)
    assert CM.doit() == 5 ** 5