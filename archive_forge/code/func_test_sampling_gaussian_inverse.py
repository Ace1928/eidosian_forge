from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.sets.sets import Interval
from sympy.external import import_module
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
from sympy.testing.pytest import skip, raises
def test_sampling_gaussian_inverse():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for sampling of Gaussian inverse.')
    X = GaussianInverse('x', 1, 1)
    assert sample(X, library='scipy') in X.pspace.domain.set