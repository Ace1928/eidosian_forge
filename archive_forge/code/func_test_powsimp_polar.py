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
def test_powsimp_polar():
    from sympy.functions.elementary.complexes import polar_lift
    from sympy.functions.elementary.exponential import exp_polar
    x, y, z = symbols('x y z')
    p, q, r = symbols('p q r', polar=True)
    assert polar_lift(-1) ** (2 * x) == exp_polar(2 * pi * I * x)
    assert powsimp(p ** x * q ** x) == (p * q) ** x
    assert p ** x * (1 / p) ** x == 1
    assert (1 / p) ** x == p ** (-x)
    assert exp_polar(x) * exp_polar(y) == exp_polar(x) * exp_polar(y)
    assert powsimp(exp_polar(x) * exp_polar(y)) == exp_polar(x + y)
    assert powsimp(exp_polar(x) * exp_polar(y) * p ** x * p ** y) == (p * exp_polar(1)) ** (x + y)
    assert powsimp(exp_polar(x) * exp_polar(y) * p ** x * p ** y, combine='exp') == exp_polar(x + y) * p ** (x + y)
    assert powsimp(exp_polar(x) * exp_polar(y) * exp_polar(2) * sin(x) + sin(y) + p ** x * p ** y) == p ** (x + y) + sin(x) * exp_polar(2 + x + y) + sin(y)
    assert powsimp(sin(exp_polar(x) * exp_polar(y))) == sin(exp_polar(x) * exp_polar(y))
    assert powsimp(sin(exp_polar(x) * exp_polar(y)), deep=True) == sin(exp_polar(x + y))