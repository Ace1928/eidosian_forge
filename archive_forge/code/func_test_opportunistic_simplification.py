import random
import concurrent.futures
from collections.abc import Hashable
from sympy.core.add import Add
from sympy.core.function import (Function, diff, expand)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.printing.str import sstr
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.matrices.matrices import (ShapeError, MatrixError,
from sympy.matrices import (
from sympy.matrices.utilities import _dotprodsimp_state
from sympy.core import Tuple, Wild
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.utilities.iterables import flatten, capture, iterable
from sympy.utilities.exceptions import ignore_warnings, SymPyDeprecationWarning
from sympy.testing.pytest import (raises, XFAIL, slow, skip, skip_under_pyodide,
from sympy.assumptions import Q
from sympy.tensor.array import Array
from sympy.matrices.expressions import MatPow
from sympy.algebras import Quaternion
from sympy.abc import a, b, c, d, x, y, z, t
def test_opportunistic_simplification():
    m = Matrix([[-5 + 5 * sqrt(2), -5], [-5 * sqrt(2) / 2 + 5, -5 * sqrt(2) / 2]])
    assert m.rank() == 1
    m = Matrix([[3 + 3 * sqrt(3) * I, -9], [4, -3 + 3 * sqrt(3) * I]])
    assert simplify(m.rref()[0] - Matrix([[1, -9 / (3 + 3 * sqrt(3) * I)], [0, 0]])) == zeros(2, 2)
    ax, ay, bx, by, cx, cy, dx, dy, ex, ey, t0, t1 = symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    m = Matrix([[ax, ay, ax * t0, ay * t0, 0], [bx, by, bx * t0, by * t0, 0], [cx, cy, cx * t0, cy * t0, 1], [dx, dy, dx * t0, dy * t0, 1], [ex, ey, 2 * ex * t1 - ex * t0, 2 * ey * t1 - ey * t0, 0]])
    assert m.rank() == 4