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
def test_transformation_to_normal():
    assert is_normal_transformation_ok(x ** 2 + 3 * y ** 2 + z ** 2 - 13 * x * y - 16 * y * z + 12 * x * z)
    assert is_normal_transformation_ok(x ** 2 + 3 * y ** 2 - 100 * z ** 2)
    assert is_normal_transformation_ok(x ** 2 + 23 * y * z)
    assert is_normal_transformation_ok(3 * y ** 2 - 100 * z ** 2 - 12 * x * y)
    assert is_normal_transformation_ok(x ** 2 + 23 * x * y - 34 * y * z + 12 * x * z)
    assert is_normal_transformation_ok(z ** 2 + 34 * x * y - 23 * y * z + x * z)
    assert is_normal_transformation_ok(x ** 2 + y ** 2 + z ** 2 - x * y - y * z - x * z)
    assert is_normal_transformation_ok(x ** 2 + 2 * y * z + 3 * z ** 2)
    assert is_normal_transformation_ok(x * y + 2 * x * z + 3 * y * z)
    assert is_normal_transformation_ok(2 * x * z + 3 * y * z)