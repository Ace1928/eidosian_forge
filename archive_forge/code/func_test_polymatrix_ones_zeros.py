from sympy.testing.pytest import raises
from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly
from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
def test_polymatrix_ones_zeros():
    assert PolyMatrix.zeros(1, 2, x) == PolyMatrix([[0, 0]], x)
    assert PolyMatrix.eye(2, x) == PolyMatrix([[1, 0], [0, 1]], x)