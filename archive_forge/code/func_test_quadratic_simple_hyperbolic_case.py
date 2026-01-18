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
def test_quadratic_simple_hyperbolic_case():
    assert diop_solve(3 * x * y + 34 * x - 12 * y + 1) == {(-133, -11), (5, -57)}
    assert diop_solve(6 * x * y + 2 * x + 3 * y + 1) == set()
    assert diop_solve(-13 * x * y + 2 * x - 4 * y - 54) == {(27, 0)}
    assert diop_solve(-27 * x * y - 30 * x - 12 * y - 54) == {(-14, -1)}
    assert diop_solve(2 * x * y + 5 * x + 56 * y + 7) == {(-161, -3), (-47, -6), (-35, -12), (-29, -69), (-27, 64), (-21, 7), (-9, 1), (105, -2)}
    assert diop_solve(6 * x * y + 9 * x + 2 * y + 3) == set()
    assert diop_solve(x * y + x + y + 1) == {(-1, t), (t, -1)}
    assert diophantine(48 * x * y)