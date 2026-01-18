from sympy.abc import x
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.utilities import (
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.testing.pytest import raises
def test_supplement_a_subspace_1():
    M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()
    B = supplement_a_subspace(M)
    assert B[:, :2] == M
    assert B[:, 2] == DomainMatrix.eye(3, QQ).to_dense()[:, 0]
    M = M.convert_to(FF(7))
    B = supplement_a_subspace(M)
    assert B[:, :2] == M
    assert B[:, 2] == DomainMatrix.eye(3, FF(7)).to_dense()[:, 1]