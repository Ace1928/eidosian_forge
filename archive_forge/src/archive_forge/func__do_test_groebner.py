from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def _do_test_groebner():
    R, x, y = ring('x,y', QQ, lex)
    f = x ** 2 + 2 * x * y ** 2
    g = x * y + 2 * y ** 3 - 1
    assert groebner([f, g], R) == [x, y ** 3 - QQ(1, 2)]
    R, y, x = ring('y,x', QQ, lex)
    f = 2 * x ** 2 * y + y ** 2
    g = 2 * x ** 3 + x * y - 1
    assert groebner([f, g], R) == [y, x ** 3 - QQ(1, 2)]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = x - z ** 2
    g = y - z ** 3
    assert groebner([f, g], R) == [f, g]
    R, x, y = ring('x,y', QQ, grlex)
    f = x ** 3 - 2 * x * y
    g = x ** 2 * y + x - 2 * y ** 2
    assert groebner([f, g], R) == [x ** 2, x * y, -QQ(1, 2) * x + y ** 2]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = -x ** 2 + y
    g = -x ** 3 + z
    assert groebner([f, g], R) == [x ** 2 - y, x * y - z, x * z - y ** 2, y ** 3 - z ** 2]
    R, x, y, z = ring('x,y,z', QQ, grlex)
    f = -x ** 2 + y
    g = -x ** 3 + z
    assert groebner([f, g], R) == [y ** 3 - z ** 2, x ** 2 - y, x * y - z, x * z - y ** 2]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = -x ** 2 + z
    g = -x ** 3 + y
    assert groebner([f, g], R) == [x ** 2 - z, x * y - z ** 2, x * z - y, y ** 2 - z ** 3]
    R, x, y, z = ring('x,y,z', QQ, grlex)
    f = -x ** 2 + z
    g = -x ** 3 + y
    assert groebner([f, g], R) == [-y ** 2 + z ** 3, x ** 2 - z, x * y - z ** 2, x * z - y]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = x - y ** 2
    g = -y ** 3 + z
    assert groebner([f, g], R) == [x - y ** 2, y ** 3 - z]
    R, x, y, z = ring('x,y,z', QQ, grlex)
    f = x - y ** 2
    g = -y ** 3 + z
    assert groebner([f, g], R) == [x ** 2 - y * z, x * y - z, -x + y ** 2]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = x - z ** 2
    g = y - z ** 3
    assert groebner([f, g], R) == [x - z ** 2, y - z ** 3]
    R, x, y, z = ring('x,y,z', QQ, grlex)
    f = x - z ** 2
    g = y - z ** 3
    assert groebner([f, g], R) == [x ** 2 - y * z, x * z - y, -x + z ** 2]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = -y ** 2 + z
    g = x - y ** 3
    assert groebner([f, g], R) == [x - y * z, y ** 2 - z]
    R, x, y, z = ring('x,y,z', QQ, grlex)
    f = -y ** 2 + z
    g = x - y ** 3
    assert groebner([f, g], R) == [-x ** 2 + z ** 3, x * y - z ** 2, y ** 2 - z, -x + y * z]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = y - z ** 2
    g = x - z ** 3
    assert groebner([f, g], R) == [x - z ** 3, y - z ** 2]
    R, x, y, z = ring('x,y,z', QQ, grlex)
    f = y - z ** 2
    g = x - z ** 3
    assert groebner([f, g], R) == [-x ** 2 + y ** 3, x * z - y ** 2, -x + y * z, -y + z ** 2]
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = 4 * x ** 2 * y ** 2 + 4 * x * y + 1
    g = x ** 2 + y ** 2 - 1
    assert groebner([f, g], R) == [x - 4 * y ** 7 + 8 * y ** 5 - 7 * y ** 3 + 3 * y, y ** 8 - 2 * y ** 6 + QQ(3, 2) * y ** 4 - QQ(1, 2) * y ** 2 + QQ(1, 16)]