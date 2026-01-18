from sympy.testing.pytest import raises
from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly
from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
def test_polymatrix_nullspace():
    M = PolyMatrix([[1, 2], [3, 6]], x)
    assert M.nullspace() == [PolyMatrix([-2, 1], x)]
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).nullspace())
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).nullspace())
    assert M.rank() == 1