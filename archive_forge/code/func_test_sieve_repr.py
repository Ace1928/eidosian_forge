from sympy.core.numbers import (I, Rational, nan, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.ntheory.generate import (sieve, Sieve)
from sympy.series.limits import limit
from sympy.ntheory import isprime, totient, mobius, randprime, nextprime, prevprime, \
from sympy.ntheory.generate import cycle_length
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises
def test_sieve_repr():
    assert 'sieve' in repr(sieve)
    assert 'prime' in repr(sieve)