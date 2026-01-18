from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.sets.sets import Interval
from sympy.external import import_module
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
from sympy.testing.pytest import skip, raises
def test_sample_scipy():
    distribs_scipy = [Beta('B', 1, 1), BetaPrime('BP', 1, 1), Cauchy('C', 1, 1), Chi('C', 1), Normal('N', 0, 1), Gamma('G', 2, 7), GammaInverse('GI', 1, 1), GaussianInverse('GUI', 1, 1), Exponential('E', 2), LogNormal('LN', 0, 1), Pareto('P', 1, 1), StudentT('S', 2), ChiSquared('CS', 2), Uniform('U', 0, 1)]
    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size, library='scipy')
            samps2 = sample(X, size=(2, 2), library='scipy')
            for sam in samps:
                assert sam in X.pspace.domain.set
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set