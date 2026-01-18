from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrtdenest3():
    z = sqrt(13 - 2 * r10 + 2 * r2 * sqrt(-2 * r10 + 11))
    assert sqrtdenest(z) == -1 + r2 + r10
    assert sqrtdenest(z, max_iter=1) == -1 + sqrt(2) + sqrt(10)
    z = sqrt(sqrt(r2 + 2) + 2)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(-2 * r10 + 4 * r2 * sqrt(-2 * r10 + 11) + 20)) == sqrt(-2 * r10 - 4 * r2 + 8 * r5 + 20)
    assert sqrtdenest(sqrt(112 + 70 * r2 + (46 + 34 * r2) * r5)) == r10 + 5 + 4 * r2 + 3 * r5
    z = sqrt(5 + sqrt(2 * r6 + 5) * sqrt(-2 * r29 + 2 * sqrt(-10 * r29 + 55) + 16))
    r = sqrt(-2 * r29 + 11)
    assert sqrtdenest(z) == sqrt(r2 * r + r3 * r + r10 + r15 + 5)
    n = sqrt(2 * r6 / 7 + 2 * r7 / 7 + 2 * sqrt(42) / 7 + 2)
    d = sqrt(16 - 2 * r29 + 2 * sqrt(55 - 10 * r29))
    assert sqrtdenest(n / d) == r7 * (1 + r6 + r7) / Mul(7, sqrt(-2 * r29 + 11) + r5, evaluate=False)