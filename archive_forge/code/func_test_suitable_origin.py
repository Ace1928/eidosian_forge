from sympy.core import symbols, Dummy, Tuple, S, Rational
from sympy.functions import hyper
from sympy.simplify.hyperexpand import Hyper_Function
def test_suitable_origin():
    assert Hyper_Function((S.Half,), (Rational(3, 2),))._is_suitable_origin() is True
    assert Hyper_Function((S.Half,), (S.Half,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (Rational(-1, 2),))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (0,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (-1, 1))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 0), (1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 1), (2, Rational(-2, 3)))._is_suitable_origin() is True
    assert Hyper_Function((S.Half, 1), (2, Rational(-2, 3), Rational(3, 2)))._is_suitable_origin() is True