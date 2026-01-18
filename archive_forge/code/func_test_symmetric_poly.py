from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises
from sympy.polys.specialpolys import (
from sympy.abc import x, y, z
def test_symmetric_poly():
    raises(ValueError, lambda: symmetric_poly(-1, x, y, z))
    raises(ValueError, lambda: symmetric_poly(5, x, y, z))
    assert symmetric_poly(1, x, y, z, polys=True) == Poly(x + y + z)
    assert symmetric_poly(1, (x, y, z), polys=True) == Poly(x + y + z)
    assert symmetric_poly(0, x, y, z) == 1
    assert symmetric_poly(1, x, y, z) == x + y + z
    assert symmetric_poly(2, x, y, z) == x * y + x * z + y * z
    assert symmetric_poly(3, x, y, z) == x * y * z