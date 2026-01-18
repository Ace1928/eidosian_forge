from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.hyperbolic import (cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.simplify.powsimp import powsimp
from sympy.simplify.fu import (
from sympy.core.random import verify_numerically
from sympy.abc import a, b, c, x, y, z
def test_process_common_addends():
    do = lambda x: Add(*[i ** (i % 2) for i in x.args])
    assert process_common_addends(Add(*[1, 2, 3, 4], evaluate=False), do, key2=lambda x: x % 2, key1=False) == 1 ** 1 + 3 ** 1 + 2 ** 0 + 4 ** 0