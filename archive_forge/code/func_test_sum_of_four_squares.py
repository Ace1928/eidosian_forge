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
def test_sum_of_four_squares():
    from sympy.core.random import randint
    n = randint(1, 100000000000000)
    assert sum((i ** 2 for i in sum_of_four_squares(n))) == n
    assert sum_of_four_squares(0) == (0, 0, 0, 0)
    assert sum_of_four_squares(14) == (0, 1, 2, 3)
    assert sum_of_four_squares(15) == (1, 1, 2, 3)
    assert sum_of_four_squares(18) == (1, 2, 2, 3)
    assert sum_of_four_squares(19) == (0, 1, 3, 3)
    assert sum_of_four_squares(48) == (0, 4, 4, 4)