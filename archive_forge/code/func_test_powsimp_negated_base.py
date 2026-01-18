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
def test_powsimp_negated_base():
    assert powsimp((-x + y) / sqrt(x - y)) == -sqrt(x - y)
    assert powsimp((-x + y) * (-z + y) / sqrt(x - y) / sqrt(z - y)) == sqrt(x - y) * sqrt(z - y)
    p = symbols('p', positive=True)
    reps = {p: 2, a: S.Half}
    assert powsimp((-p) ** a / p ** a).subs(reps) == ((-1) ** a).subs(reps)
    assert powsimp((-p) ** a * p ** a).subs(reps) == ((-p ** 2) ** a).subs(reps)
    n = symbols('n', negative=True)
    reps = {p: -2, a: S.Half}
    assert powsimp((-n) ** a / n ** a).subs(reps) == (-1) ** (-a).subs(a, S.Half)
    assert powsimp((-n) ** a * n ** a).subs(reps) == ((-n ** 2) ** a).subs(reps)
    eq = (-x) ** a / x ** a
    assert powsimp(eq) == eq