from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrtdenest4():
    z = sqrt(8 - r2 * sqrt(5 - r5) - sqrt(3) * (1 + r5))
    z1 = sqrtdenest(z)
    c = sqrt(-r5 + 5)
    z1 = ((-r15 * c - r3 * c + c + r5 * c - r6 - r2 + r10 + sqrt(30)) / 4).expand()
    assert sqrtdenest(z) == z1
    z = sqrt(2 * r2 * sqrt(r2 + 2) + 5 * r2 + 4 * sqrt(r2 + 2) + 8)
    assert sqrtdenest(z) == r2 + sqrt(r2 + 2) + 2
    w = 2 + r2 + r3 + (1 + r3) * sqrt(2 + r2 + 5 * r3)
    z = sqrt((w ** 2).expand())
    assert sqrtdenest(z) == w.expand()