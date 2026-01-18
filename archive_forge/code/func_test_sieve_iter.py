from sympy.core.numbers import (I, Rational, nan, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.ntheory.generate import (sieve, Sieve)
from sympy.series.limits import limit
from sympy.ntheory import isprime, totient, mobius, randprime, nextprime, prevprime, \
from sympy.ntheory.generate import cycle_length
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises
def test_sieve_iter():
    values = []
    for value in sieve:
        if value > 7:
            break
        values.append(value)
    assert values == list(sieve[1:5])