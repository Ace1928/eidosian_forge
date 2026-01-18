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
def test_proper_divisors_and_proper_divisor_count():
    assert proper_divisors(-1) == []
    assert proper_divisors(0) == []
    assert proper_divisors(1) == []
    assert proper_divisors(2) == [1]
    assert proper_divisors(3) == [1]
    assert proper_divisors(17) == [1]
    assert proper_divisors(10) == [1, 2, 5]
    assert proper_divisors(100) == [1, 2, 4, 5, 10, 20, 25, 50]
    assert proper_divisors(1000000007) == [1]
    assert proper_divisor_count(0) == 0
    assert proper_divisor_count(-1) == 0
    assert proper_divisor_count(1) == 0
    assert proper_divisor_count(36) == 8
    assert proper_divisor_count(2 * 3 * 5) == 7