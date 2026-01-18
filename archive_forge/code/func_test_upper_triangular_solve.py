from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_upper_triangular_solve():
    raises(NonSquareMatrixError, lambda: SparseMatrix([[1, 2]]).upper_triangular_solve(Matrix([[1, 2]])))
    raises(ShapeError, lambda: SparseMatrix([[1, 2], [0, 4]]).upper_triangular_solve(Matrix([1])))
    raises(TypeError, lambda: SparseMatrix([[1, 2], [3, 4]]).upper_triangular_solve(Matrix([[1, 2], [3, 4]])))
    a, b, c, d = symbols('a:d')
    u, v, w, x = symbols('u:x')
    A = SparseMatrix([[a, b], [0, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])
    sol = Matrix([[(u - b * w / d) / a, (v - b * x / d) / a], [w / d, x / d]])
    assert A.upper_triangular_solve(B) == sol
    assert A.upper_triangular_solve(C) == sol