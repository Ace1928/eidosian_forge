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
def test_udivisor_sigma():
    assert [udivisor_sigma(k) for k in range(1, 12)] == [1, 3, 4, 5, 6, 12, 8, 9, 10, 18, 12]
    assert [udivisor_sigma(k, 3) for k in range(1, 12)] == [1, 9, 28, 65, 126, 252, 344, 513, 730, 1134, 1332]
    assert udivisor_sigma(23450) == 42432
    assert udivisor_sigma(23450, 0) == 16
    assert udivisor_sigma(23450, 1) == 42432
    assert udivisor_sigma(23450, 2) == 702685000
    assert udivisor_sigma(23450, 4) == 321426961814978248
    m = Symbol('m', integer=True)
    k = Symbol('k', integer=True)
    assert udivisor_sigma(m)
    assert udivisor_sigma(m, k)
    assert udivisor_sigma(m).subs(m, 4 ** 9) == 262145
    assert udivisor_sigma(m, k).subs([(m, 4 ** 9), (k, 2)]) == 68719476737
    assert summation(udivisor_sigma(m), (m, 2, 15)) == 169