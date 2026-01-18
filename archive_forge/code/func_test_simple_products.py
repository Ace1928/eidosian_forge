from math import prod
from sympy.concrete.expr_with_intlimits import ReorderError
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import (Sum, summation, telescopic,
from sympy.core.function import (Derivative, Function)
from sympy.core import (Catalan, EulerGamma)
from sympy.core.facts import InconsistentAssumptions
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Or
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices import (Matrix, SparseMatrix,
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.abc import a, b, c, d, k, m, x, y, z
def test_simple_products():
    assert Product(S.NaN, (x, 1, 3)) is S.NaN
    assert product(S.NaN, (x, 1, 3)) is S.NaN
    assert Product(x, (n, a, a)).doit() == x
    assert Product(x, (x, a, a)).doit() == a
    assert Product(x, (y, 1, a)).doit() == x ** a
    lo, hi = (1, 2)
    s1 = Product(n, (n, lo, hi))
    s2 = Product(n, (n, hi, lo))
    assert s1 != s2
    assert s1.doit() == 2
    assert s2.doit() == 1
    lo, hi = (x, x + 1)
    s1 = Product(n, (n, lo, hi))
    s2 = Product(n, (n, hi, lo))
    s3 = 1 / Product(n, (n, hi + 1, lo - 1))
    assert s1 != s2
    assert s1.doit() == x * (x + 1)
    assert s2.doit() == 1
    assert s3.doit() == x * (x + 1)
    assert Product(Integral(2 * x, (x, 1, y)) + 2 * x, (x, 1, 2)).doit() == (y ** 2 + 1) * (y ** 2 + 3)
    assert product(2, (n, a, b)) == 2 ** (b - a + 1)
    assert product(n, (n, 1, b)) == factorial(b)
    assert product(n ** 3, (n, 1, b)) == factorial(b) ** 3
    assert product(3 ** (2 + n), (n, a, b)) == 3 ** (2 * (1 - a + b) + b / 2 + b ** 2 / 2 + a / 2 - a ** 2 / 2)
    assert product(cos(n), (n, 3, 5)) == cos(3) * cos(4) * cos(5)
    assert product(cos(n), (n, x, x + 2)) == cos(x) * cos(x + 1) * cos(x + 2)
    assert isinstance(product(cos(n), (n, x, x + S.Half)), Product)
    assert isinstance(Product(n ** n, (n, 1, b)), Product)