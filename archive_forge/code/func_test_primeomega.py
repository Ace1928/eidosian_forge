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
def test_primeomega():
    assert primeomega(2) == 1
    assert primeomega(2 * 2) == 2
    assert primeomega(2 * 2 * 3) == 3
    assert primeomega(3 * 25) == primeomega(3) + primeomega(25)
    assert [primeomega(p) for p in primerange(1, 10)] == [1, 1, 1, 1]
    assert primeomega(fac(50)) == 108
    assert primeomega(2 ** 9941 - 1) == 1
    n = Symbol('n', integer=True)
    assert primeomega(n)
    assert primeomega(n).subs(n, 2 ** 31 - 1) == 1
    assert summation(primeomega(n), (n, 2, 30)) == 59