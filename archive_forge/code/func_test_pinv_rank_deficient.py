from sympy.core.function import expand_mul
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.matrices.matrices import (ShapeError, NonSquareMatrixError)
from sympy.matrices import (
from sympy.testing.pytest import raises
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.solvers.solveset import linsolve
from sympy.abc import x, y
def test_pinv_rank_deficient():
    As = [Matrix([[1, 1, 1], [2, 2, 2]]), Matrix([[1, 0], [0, 0]]), Matrix([[1, 2], [2, 4], [3, 6]])]
    for A in As:
        A_pinv = A.pinv(method='RD')
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA
    for A in As:
        A_pinv = A.pinv(method='ED')
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA
    A = Matrix([[1, 0], [0, 0]])
    B = Matrix([3, 0])
    solution = A.pinv_solve(B)
    w1 = solution.atoms(Symbol).pop()
    assert w1.name == 'w1_0'
    assert solution == Matrix([3, w1])
    assert A * A.pinv() * B == B
    B = Matrix([3, 1])
    solution = A.pinv_solve(B)
    w1 = solution.atoms(Symbol).pop()
    assert w1.name == 'w1_0'
    assert solution == Matrix([3, w1])
    assert A * A.pinv() * B != B