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
def test_telescopic_sums():
    assert Sum(1 / k - 1 / (k + 1), (k, 1, n)).doit() == 1 - 1 / (1 + n)
    assert Sum(f(k) - f(k + 2), (k, m, n)).doit() == -f(1 + n) - f(2 + n) + f(m) + f(1 + m)
    assert Sum(cos(k) - cos(k + 3), (k, 1, n)).doit() == -cos(1 + n) - cos(2 + n) - cos(3 + n) + cos(1) + cos(2) + cos(3)
    assert telescopic(1 / m, -m / (1 + m), (m, n - 1, n)) == telescopic(1 / k, -k / (1 + k), (k, n - 1, n))
    assert Sum(1 / x / (x - 1), (x, a, b)).doit() == 1 / (a - 1) - 1 / b
    eq = 1 / ((5 * n + 2) * (5 * (n + 1) + 2))
    assert Sum(eq, (n, 0, oo)).doit() == S(1) / 10
    nz = symbols('nz', nonzero=True)
    v = Sum(eq.subs(5, nz), (n, 0, oo)).doit()
    assert v.subs(nz, 5).simplify() == S(1) / 10
    s = Sum(eq, (n, 0, k)).doit()
    v = Sum(eq, (n, 0, 10 ** 100)).doit()
    assert v == s.subs(k, 10 ** 100)