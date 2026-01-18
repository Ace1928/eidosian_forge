from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_representing_matrices():
    R, x, y = ring('x,y', QQ, grlex)
    basis = [(0, 0), (0, 1), (1, 0), (1, 1)]
    F = [x ** 2 - x - 3 * y + 1, -2 * x + y ** 2 + y - 1]
    assert _representing_matrices(basis, F, R) == [[[QQ(0, 1), QQ(0, 1), -QQ(1, 1), QQ(3, 1)], [QQ(0, 1), QQ(0, 1), QQ(3, 1), -QQ(4, 1)], [QQ(1, 1), QQ(0, 1), QQ(1, 1), QQ(6, 1)], [QQ(0, 1), QQ(1, 1), QQ(0, 1), QQ(1, 1)]], [[QQ(0, 1), QQ(1, 1), QQ(0, 1), -QQ(2, 1)], [QQ(1, 1), -QQ(1, 1), QQ(0, 1), QQ(6, 1)], [QQ(0, 1), QQ(2, 1), QQ(0, 1), QQ(3, 1)], [QQ(0, 1), QQ(0, 1), QQ(1, 1), -QQ(1, 1)]]]