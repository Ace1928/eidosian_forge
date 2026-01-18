from sympy.assumptions.refine import refine
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
from sympy.core.function import (Function, expand, WildFunction,
from sympy.core.mul import Mul
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Lt, Gt, Le
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import tan, sin, cos
from sympy.functions.special.delta_functions import (Heaviside,
from sympy.functions.special.error_functions import Si
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate, Integral
from sympy.physics.secondquant import FockState
from sympy.polys.partfrac import apart
from sympy.polys.polytools import factor, cancel, Poly
from sympy.polys.rationaltools import together
from sympy.series.order import O
from sympy.sets.sets import FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import collect, radsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify, nsimplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import Indexed
from sympy.physics.units import meter
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import a, b, c, n, t, u, x, y, z
def test_relational_assumptions():
    m1 = Symbol('m1', nonnegative=False)
    m2 = Symbol('m2', positive=False)
    m3 = Symbol('m3', nonpositive=False)
    m4 = Symbol('m4', negative=False)
    assert (m1 < 0) == Lt(m1, 0)
    assert (m2 <= 0) == Le(m2, 0)
    assert (m3 > 0) == Gt(m3, 0)
    assert (m4 >= 0) == Ge(m4, 0)
    m1 = Symbol('m1', nonnegative=False, real=True)
    m2 = Symbol('m2', positive=False, real=True)
    m3 = Symbol('m3', nonpositive=False, real=True)
    m4 = Symbol('m4', negative=False, real=True)
    assert (m1 < 0) is S.true
    assert (m2 <= 0) is S.true
    assert (m3 > 0) is S.true
    assert (m4 >= 0) is S.true
    m1 = Symbol('m1', negative=True)
    m2 = Symbol('m2', nonpositive=True)
    m3 = Symbol('m3', positive=True)
    m4 = Symbol('m4', nonnegative=True)
    assert (m1 < 0) is S.true
    assert (m2 <= 0) is S.true
    assert (m3 > 0) is S.true
    assert (m4 >= 0) is S.true
    m1 = Symbol('m1', negative=False, real=True)
    m2 = Symbol('m2', nonpositive=False, real=True)
    m3 = Symbol('m3', positive=False, real=True)
    m4 = Symbol('m4', nonnegative=False, real=True)
    assert (m1 < 0) is S.false
    assert (m2 <= 0) is S.false
    assert (m3 > 0) is S.false
    assert (m4 >= 0) is S.false