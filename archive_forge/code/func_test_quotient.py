from sympy.polys import QQ, ilex
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_quotient():
    R = QQ.old_poly_ring(x, y, z)
    assert R.ideal(x, y).quotient(R.ideal(y ** 2, z)) == R.ideal(x, y)