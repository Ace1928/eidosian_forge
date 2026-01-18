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
def test_quadratic_non_perfect_square():
    assert check_solutions(x ** 2 - 2 * x - 5 * y ** 2)
    assert check_solutions(3 * x ** 2 - 2 * y ** 2 - 2 * x - 2 * y)
    assert check_solutions(x ** 2 - x * y - y ** 2 - 3 * y)
    assert check_solutions(x ** 2 - 9 * y ** 2 - 2 * x - 6 * y)
    assert BinaryQuadratic(x ** 2 + y ** 2 + 2 * x + 2 * y + 2).solve() == {(-1, -1)}