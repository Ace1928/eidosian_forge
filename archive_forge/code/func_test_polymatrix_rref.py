from sympy.testing.pytest import raises
from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly
from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
def test_polymatrix_rref():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M.rref() == (PolyMatrix.eye(2, x), (0, 1))
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).rref())
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).rref())