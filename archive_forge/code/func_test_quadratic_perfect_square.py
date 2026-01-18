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
def test_quadratic_perfect_square():
    assert check_solutions(48 * x * y)
    assert check_solutions(4 * x ** 2 - 5 * x * y + y ** 2 + 2)
    assert check_solutions(-2 * x ** 2 - 3 * x * y + 2 * y ** 2 - 2 * x - 17 * y + 25)
    assert check_solutions(12 * x ** 2 + 13 * x * y + 3 * y ** 2 - 2 * x + 3 * y - 12)
    assert check_solutions(8 * x ** 2 + 10 * x * y + 2 * y ** 2 - 32 * x - 13 * y - 23)
    assert check_solutions(4 * x ** 2 - 4 * x * y - 3 * y - 8 * x - 3)
    assert check_solutions(-4 * x * y - 4 * y ** 2 - 3 * y - 5 * x - 10)
    assert check_solutions(x ** 2 - y ** 2 - 2 * x - 2 * y)
    assert check_solutions(x ** 2 - 9 * y ** 2 - 2 * x - 6 * y)
    assert check_solutions(4 * x ** 2 - 9 * y ** 2 - 4 * x - 12 * y - 3)