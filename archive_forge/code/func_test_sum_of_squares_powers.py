from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
def test_sum_of_squares_powers():
    tru = {(0, 0, 1, 1, 11), (0, 0, 5, 7, 7), (0, 1, 3, 7, 8), (0, 1, 4, 5, 9), (0, 3, 4, 7, 7), (0, 3, 5, 5, 8), (1, 1, 2, 6, 9), (1, 1, 6, 6, 7), (1, 2, 3, 3, 10), (1, 3, 4, 4, 9), (1, 5, 5, 6, 6), (2, 2, 3, 5, 9), (2, 3, 5, 6, 7), (3, 3, 4, 5, 8)}
    eq = u ** 2 + v ** 2 + x ** 2 + y ** 2 + z ** 2 - 123
    ans = diop_general_sum_of_squares(eq, oo)
    assert len(ans) == 14
    assert ans == tru
    raises(ValueError, lambda: list(sum_of_squares(10, -1)))
    assert list(sum_of_squares(-10, 2)) == []
    assert list(sum_of_squares(2, 3)) == []
    assert list(sum_of_squares(0, 3, True)) == [(0, 0, 0)]
    assert list(sum_of_squares(0, 3)) == []
    assert list(sum_of_squares(4, 1)) == [(2,)]
    assert list(sum_of_squares(5, 1)) == []
    assert list(sum_of_squares(50, 2)) == [(5, 5), (1, 7)]
    assert list(sum_of_squares(11, 5, True)) == [(1, 1, 1, 2, 2), (0, 0, 1, 1, 3)]
    assert list(sum_of_squares(8, 8)) == [(1, 1, 1, 1, 1, 1, 1, 1)]
    assert [len(list(sum_of_squares(i, 5, True))) for i in range(30)] == [1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 3, 2, 1, 3, 3, 3, 3, 4, 3, 3, 2, 2, 4, 4, 4, 4, 5]
    assert [len(list(sum_of_squares(i, 5))) for i in range(30)] == [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3]
    for i in range(30):
        s1 = set(sum_of_squares(i, 5, True))
        assert not s1 or all((sum((j ** 2 for j in t)) == i for t in s1))
        s2 = set(sum_of_squares(i, 5))
        assert all((sum((j ** 2 for j in t)) == i for t in s2))
    raises(ValueError, lambda: list(sum_of_powers(2, -1, 1)))
    raises(ValueError, lambda: list(sum_of_powers(2, 1, -1)))
    assert list(sum_of_powers(-2, 3, 2)) == [(-1, -1)]
    assert list(sum_of_powers(-2, 4, 2)) == []
    assert list(sum_of_powers(2, 1, 1)) == [(2,)]
    assert list(sum_of_powers(2, 1, 3, True)) == [(0, 0, 2), (0, 1, 1)]
    assert list(sum_of_powers(5, 1, 2, True)) == [(0, 5), (1, 4), (2, 3)]
    assert list(sum_of_powers(6, 2, 2)) == []
    assert list(sum_of_powers(3 ** 5, 3, 1)) == []
    assert list(sum_of_powers(3 ** 6, 3, 1)) == [(9,)] and 9 ** 3 == 3 ** 6
    assert list(sum_of_powers(2 ** 1000, 5, 2)) == []