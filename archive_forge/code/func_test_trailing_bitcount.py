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
def test_trailing_bitcount():
    assert trailing(0) == 0
    assert trailing(1) == 0
    assert trailing(-1) == 0
    assert trailing(2) == 1
    assert trailing(7) == 0
    assert trailing(-7) == 0
    for i in range(100):
        assert trailing(1 << i) == i
        assert trailing((1 << i) * 31337) == i
    assert trailing(1 << 1000001) == 1000001
    assert trailing((1 << 273956) * 7 ** 37) == 273956
    big = small_trailing[-1] * 2
    assert trailing(-big) == trailing(big)
    assert bitcount(-big) == bitcount(big)