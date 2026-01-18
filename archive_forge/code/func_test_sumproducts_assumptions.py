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
def test_sumproducts_assumptions():
    M = Symbol('M', integer=True, positive=True)
    m = Symbol('m', integer=True)
    for func in [Sum, Product]:
        assert func(m, (m, -M, M)).is_positive is None
        assert func(m, (m, -M, M)).is_nonpositive is None
        assert func(m, (m, -M, M)).is_negative is None
        assert func(m, (m, -M, M)).is_nonnegative is None
        assert func(m, (m, -M, M)).is_finite is True
    m = Symbol('m', integer=True, nonnegative=True)
    for func in [Sum, Product]:
        assert func(m, (m, 0, M)).is_positive is None
        assert func(m, (m, 0, M)).is_nonpositive is None
        assert func(m, (m, 0, M)).is_negative is False
        assert func(m, (m, 0, M)).is_nonnegative is True
        assert func(m, (m, 0, M)).is_finite is True
    m = Symbol('m', integer=True, positive=True)
    for func in [Sum, Product]:
        assert func(m, (m, 1, M)).is_positive is True
        assert func(m, (m, 1, M)).is_nonpositive is False
        assert func(m, (m, 1, M)).is_negative is False
        assert func(m, (m, 1, M)).is_nonnegative is True
        assert func(m, (m, 1, M)).is_finite is True
    m = Symbol('m', integer=True, negative=True)
    assert Sum(m, (m, -M, -1)).is_positive is False
    assert Sum(m, (m, -M, -1)).is_nonpositive is True
    assert Sum(m, (m, -M, -1)).is_negative is True
    assert Sum(m, (m, -M, -1)).is_nonnegative is False
    assert Sum(m, (m, -M, -1)).is_finite is True
    assert Product(m, (m, -M, -1)).is_positive is None
    assert Product(m, (m, -M, -1)).is_nonpositive is None
    assert Product(m, (m, -M, -1)).is_negative is None
    assert Product(m, (m, -M, -1)).is_nonnegative is None
    assert Product(m, (m, -M, -1)).is_finite is True
    m = Symbol('m', integer=True, nonpositive=True)
    assert Sum(m, (m, -M, 0)).is_positive is False
    assert Sum(m, (m, -M, 0)).is_nonpositive is True
    assert Sum(m, (m, -M, 0)).is_negative is None
    assert Sum(m, (m, -M, 0)).is_nonnegative is None
    assert Sum(m, (m, -M, 0)).is_finite is True
    assert Product(m, (m, -M, 0)).is_positive is None
    assert Product(m, (m, -M, 0)).is_nonpositive is None
    assert Product(m, (m, -M, 0)).is_negative is None
    assert Product(m, (m, -M, 0)).is_nonnegative is None
    assert Product(m, (m, -M, 0)).is_finite is True
    m = Symbol('m', integer=True)
    assert Sum(2, (m, 0, oo)).is_positive is None
    assert Sum(2, (m, 0, oo)).is_nonpositive is None
    assert Sum(2, (m, 0, oo)).is_negative is None
    assert Sum(2, (m, 0, oo)).is_nonnegative is None
    assert Sum(2, (m, 0, oo)).is_finite is None
    assert Product(2, (m, 0, oo)).is_positive is None
    assert Product(2, (m, 0, oo)).is_nonpositive is None
    assert Product(2, (m, 0, oo)).is_negative is False
    assert Product(2, (m, 0, oo)).is_nonnegative is None
    assert Product(2, (m, 0, oo)).is_finite is None
    assert Product(0, (x, M, M - 1)).is_positive is True
    assert Product(0, (x, M, M - 1)).is_finite is True