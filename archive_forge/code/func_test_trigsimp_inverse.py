from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)
from sympy.testing.pytest import XFAIL
from sympy.abc import x, y
def test_trigsimp_inverse():
    alpha = symbols('alpha')
    s, c = (sin(alpha), cos(alpha))
    for finv in [asin, acos, asec, acsc, atan, acot]:
        f = finv.inverse(None)
        assert alpha == trigsimp(finv(f(alpha)), inverse=True)
    for a, b in [[c, s], [s, c]]:
        for i, j in product([-1, 1], repeat=2):
            angle = atan2(i * b, j * a)
            angle_inverted = trigsimp(angle, inverse=True)
            assert angle_inverted != angle
            assert sin(angle_inverted) == trigsimp(sin(angle))
            assert cos(angle_inverted) == trigsimp(cos(angle))