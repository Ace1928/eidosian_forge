from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrt_symbolic_denest():
    x = Symbol('x')
    z = sqrt(((1 + sqrt(sqrt(2 + x) + 3)) ** 2).expand())
    assert sqrtdenest(z) == sqrt((1 + sqrt(sqrt(2 + x) + 3)) ** 2)
    z = sqrt(((1 + sqrt(sqrt(2 + cos(1)) + 3)) ** 2).expand())
    assert sqrtdenest(z) == 1 + sqrt(sqrt(2 + cos(1)) + 3)
    z = ((1 + cos(2)) ** 4 + 1).expand()
    assert sqrtdenest(z) == z
    z = sqrt(((1 + sqrt(sqrt(2 + cos(3 * x)) + 3)) ** 2 + 1).expand())
    assert sqrtdenest(z) == z
    c = cos(3)
    c2 = c ** 2
    assert sqrtdenest(sqrt(2 * sqrt(1 + r3) * c + c2 + 1 + r3 * c2)) == -1 - sqrt(1 + r3) * c
    ra = sqrt(1 + r3)
    z = sqrt(20 * ra * sqrt(3 + 3 * r3) + 12 * r3 * ra * sqrt(3 + 3 * r3) + 64 * r3 + 112)
    assert sqrtdenest(z) == z