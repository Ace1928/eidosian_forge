from sympy.testing.pytest import raises
from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly
from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
def test_polymatrix_getitem():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M[:, :] == M
    assert M[0, :] == PolyMatrix([[1, 2]], x)
    assert M[:, 0] == PolyMatrix([1, 3], x)
    assert M[0, 0] == Poly(1, x, domain=QQ)
    assert M[0] == Poly(1, x, domain=QQ)
    assert M[:2] == [Poly(1, x, domain=QQ), Poly(2, x, domain=QQ)]