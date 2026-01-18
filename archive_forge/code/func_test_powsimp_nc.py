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
def test_powsimp_nc():
    x, y, z = symbols('x,y,z')
    A, B, C = symbols('A B C', commutative=False)
    assert powsimp(A ** x * A ** y, combine='all') == A ** (x + y)
    assert powsimp(A ** x * A ** y, combine='base') == A ** x * A ** y
    assert powsimp(A ** x * A ** y, combine='exp') == A ** (x + y)
    assert powsimp(A ** x * B ** x, combine='all') == A ** x * B ** x
    assert powsimp(A ** x * B ** x, combine='base') == A ** x * B ** x
    assert powsimp(A ** x * B ** x, combine='exp') == A ** x * B ** x
    assert powsimp(B ** x * A ** x, combine='all') == B ** x * A ** x
    assert powsimp(B ** x * A ** x, combine='base') == B ** x * A ** x
    assert powsimp(B ** x * A ** x, combine='exp') == B ** x * A ** x
    assert powsimp(A ** x * A ** y * A ** z, combine='all') == A ** (x + y + z)
    assert powsimp(A ** x * A ** y * A ** z, combine='base') == A ** x * A ** y * A ** z
    assert powsimp(A ** x * A ** y * A ** z, combine='exp') == A ** (x + y + z)
    assert powsimp(A ** x * B ** x * C ** x, combine='all') == A ** x * B ** x * C ** x
    assert powsimp(A ** x * B ** x * C ** x, combine='base') == A ** x * B ** x * C ** x
    assert powsimp(A ** x * B ** x * C ** x, combine='exp') == A ** x * B ** x * C ** x
    assert powsimp(B ** x * A ** x * C ** x, combine='all') == B ** x * A ** x * C ** x
    assert powsimp(B ** x * A ** x * C ** x, combine='base') == B ** x * A ** x * C ** x
    assert powsimp(B ** x * A ** x * C ** x, combine='exp') == B ** x * A ** x * C ** x