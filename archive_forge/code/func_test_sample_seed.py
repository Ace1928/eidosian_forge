from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.external import import_module
from sympy.stats import Binomial, sample, Die, FiniteRV, DiscreteUniform, Bernoulli, BetaBinomial, Hypergeometric, \
from sympy.testing.pytest import skip, raises
def test_sample_seed():
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)})
    size = 10
    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0 = sample(F, size=size, library=lib, seed=0)
                s1 = sample(F, size=size, library=lib, seed=0)
                s2 = sample(F, size=size, library=lib, seed=1)
                assert all(s0 == s1)
                assert not all(s1 == s2)
        except NotImplementedError:
            continue