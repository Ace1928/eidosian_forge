from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test_simplify_prod():
    y, t, b, c, v, d = symbols('y, t, b, c, v, d', integer=True)
    _simplify = lambda e: simplify(e, doit=False)
    assert _simplify(Product(x * y, (x, n, m), (y, a, k)) * Product(y, (x, n, m), (y, a, k))) == Product(x * y ** 2, (x, n, m), (y, a, k))
    assert _simplify(3 * y * Product(x, (x, n, m)) * Product(x, (x, m + 1, a))) == 3 * y * Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x, (x, n, k))) == Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))) == Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))
    assert _simplify(Product(x, (t, a, b)) * Product(y, (t, a, b)) * Product(x, (t, b + 1, c))) == Product(x * y, (t, a, b)) * Product(x, (t, b + 1, c))
    assert _simplify(Product(x, (t, a, b)) * Product(x, (t, b + 1, c)) * Product(y, (t, a, b))) == Product(x * y, (t, a, b)) * Product(x, (t, b + 1, c))
    assert _simplify(Product(sin(t) ** 2 + cos(t) ** 2 + 1, (t, a, b))) == Product(2, (t, a, b))
    assert _simplify(Product(sin(t) ** 2 + cos(t) ** 2 - 1, (t, a, b))) == Product(0, (t, a, b))
    assert _simplify(Product(v * Product(sin(t) ** 2 + cos(t) ** 2, (t, a, b)), (v, c, d))) == Product(v * Product(1, (t, a, b)), (v, c, d))