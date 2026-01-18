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
def test_special_products():
    assert product((4 * k) ** 2 / (4 * k ** 2 - 1), (k, 1, n)) == 4 ** n * factorial(n) ** 2 / rf(S.Half, n) / rf(Rational(3, 2), n)
    assert product(1 + a / k ** 2, (k, 1, n)) == rf(1 - sqrt(-a), n) * rf(1 + sqrt(-a), n) / factorial(n) ** 2