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
def test_quadratic_parabolic_case():
    assert check_solutions(8 * x ** 2 - 24 * x * y + 18 * y ** 2 + 5 * x + 7 * y + 16)
    assert check_solutions(8 * x ** 2 - 24 * x * y + 18 * y ** 2 + 6 * x + 12 * y - 6)
    assert check_solutions(8 * x ** 2 + 24 * x * y + 18 * y ** 2 + 4 * x + 6 * y - 7)
    assert check_solutions(-4 * x ** 2 + 4 * x * y - y ** 2 + 2 * x - 3)
    assert check_solutions(x ** 2 + 2 * x * y + y ** 2 + 2 * x + 2 * y + 1)
    assert check_solutions(x ** 2 - 2 * x * y + y ** 2 + 2 * x + 2 * y + 1)
    assert check_solutions(y ** 2 - 41 * x + 40)