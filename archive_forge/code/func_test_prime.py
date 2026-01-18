from sympy.core.numbers import (I, Rational, nan, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.ntheory.generate import (sieve, Sieve)
from sympy.series.limits import limit
from sympy.ntheory import isprime, totient, mobius, randprime, nextprime, prevprime, \
from sympy.ntheory.generate import cycle_length
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises
def test_prime():
    assert prime(1) == 2
    assert prime(2) == 3
    assert prime(5) == 11
    assert prime(11) == 31
    assert prime(57) == 269
    assert prime(296) == 1949
    assert prime(559) == 4051
    assert prime(3000) == 27449
    assert prime(4096) == 38873
    assert prime(9096) == 94321
    assert prime(25023) == 287341
    assert prime(10000000) == 179424673
    assert prime(99999999) == 2038074739
    raises(ValueError, lambda: prime(0))
    sieve.extend(3000)
    assert prime(401) == 2749
    raises(ValueError, lambda: prime(-1))