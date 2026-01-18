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
def test_wronskian():
    assert wronskian([cos(x), sin(x)], x) == cos(x) ** 2 + sin(x) ** 2
    assert wronskian([exp(x), exp(2 * x)], x) == exp(3 * x)
    assert wronskian([exp(x), x], x) == exp(x) - x * exp(x)
    assert wronskian([1, x, x ** 2], x) == 2
    w1 = -6 * exp(x) * sin(x) * x + 6 * cos(x) * exp(x) * x ** 2 - 6 * exp(x) * cos(x) * x - exp(x) * cos(x) * x ** 3 + exp(x) * sin(x) * x ** 3
    assert wronskian([exp(x), cos(x), x ** 3], x).expand() == w1
    assert wronskian([exp(x), cos(x), x ** 3], x, method='berkowitz').expand() == w1
    w2 = -x ** 3 * cos(x) ** 2 - x ** 3 * sin(x) ** 2 - 6 * x * cos(x) ** 2 - 6 * x * sin(x) ** 2
    assert wronskian([sin(x), cos(x), x ** 3], x).expand() == w2
    assert wronskian([sin(x), cos(x), x ** 3], x, method='berkowitz').expand() == w2
    assert wronskian([], x) == 1