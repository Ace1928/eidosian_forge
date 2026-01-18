from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_special_constants():
    assert S.Zero == Integer(0)
    assert S.One == Integer(1)
    assert S.NegativeOne == Integer(-1)
    assert S.Half == Rational(1, 2)