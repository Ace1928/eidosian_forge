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
def test_parametrize_ternary_quadratic():
    assert check_solutions(x ** 2 + y ** 2 - z ** 2)
    assert check_solutions(x ** 2 + 2 * x * y + z ** 2)
    assert check_solutions(234 * x ** 2 - 65601 * y ** 2 - z ** 2)
    assert check_solutions(3 * x ** 2 + 2 * y ** 2 - z ** 2 - 2 * x * y + 5 * y * z - 7 * y * z)
    assert check_solutions(x ** 2 - y ** 2 - z ** 2)
    assert check_solutions(x ** 2 - 49 * y ** 2 - z ** 2 + 13 * z * y - 8 * x * y)
    assert check_solutions(8 * x * y + z ** 2)
    assert check_solutions(124 * x ** 2 - 30 * y ** 2 - 7729 * z ** 2)
    assert check_solutions(236 * x ** 2 - 225 * y ** 2 - 11 * x * y - 13 * y * z - 17 * x * z)
    assert check_solutions(90 * x ** 2 + 3 * y ** 2 + 5 * x * y + 2 * z * y + 5 * x * z)
    assert check_solutions(124 * x ** 2 - 30 * y ** 2 - 7729 * z ** 2)