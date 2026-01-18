from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.powsimp import (powdenest, powsimp)
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.core.symbol import Str
from sympy.abc import x, y, z, a, b
def test_powdenest_polar():
    x, y, z = symbols('x y z', polar=True)
    a, b, c = symbols('a b c')
    assert powdenest((x * y * z) ** a) == x ** a * y ** a * z ** a
    assert powdenest((x ** a * y ** b) ** c) == x ** (a * c) * y ** (b * c)
    assert powdenest(((x ** a) ** b * y ** c) ** c) == x ** (a * b * c) * y ** c ** 2