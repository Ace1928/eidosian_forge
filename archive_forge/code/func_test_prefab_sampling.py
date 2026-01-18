from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.sets.sets import Interval
from sympy.external import import_module
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
from sympy.testing.pytest import skip, raises
def test_prefab_sampling():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    N = Normal('X', 0, 1)
    L = LogNormal('L', 0, 1)
    E = Exponential('Ex', 1)
    P = Pareto('P', 1, 3)
    W = Weibull('W', 1, 1)
    U = Uniform('U', 0, 1)
    B = Beta('B', 2, 5)
    G = Gamma('G', 1, 3)
    variables = [N, L, E, P, W, U, B, G]
    niter = 10
    size = 5
    for var in variables:
        for _ in range(niter):
            assert sample(var) in var.pspace.domain.set
            samps = sample(var, size=size)
            for samp in samps:
                assert samp in var.pspace.domain.set