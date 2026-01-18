from sympy.concrete.summations import summation
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial as fac
from sympy.core.evalf import bitcount
from sympy.core.numbers import Integer, Rational
from sympy.ntheory import (totient,
from sympy.ntheory.factor_ import (smoothness, smoothness_p, proper_divisors,
from sympy.testing.pytest import raises, slow
from sympy.utilities.iterables import capture
def test_primenu():
    assert primenu(2) == 1
    assert primenu(2 * 3) == 2
    assert primenu(2 * 3 * 5) == 3
    assert primenu(3 * 25) == primenu(3) + primenu(25)
    assert [primenu(p) for p in primerange(1, 10)] == [1, 1, 1, 1]
    assert primenu(fac(50)) == 15
    assert primenu(2 ** 9941 - 1) == 1
    n = Symbol('n', integer=True)
    assert primenu(n)
    assert primenu(n).subs(n, 2 ** 31 - 1) == 1
    assert summation(primenu(n), (n, 2, 30)) == 43