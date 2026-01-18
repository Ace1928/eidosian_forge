from math import isclose
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.functions.special.error_functions import (erf, erfc,
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm
from sympy.solvers import solve
from sympy.solvers.solveset import (
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
@XFAIL
def test_solve_lambert():
    assert solveset_real(x * exp(x) - 1, x) == FiniteSet(LambertW(1))
    assert solveset_real(exp(x) + x, x) == FiniteSet(-LambertW(1))
    assert solveset_real(x + 2 ** x, x) == FiniteSet(-LambertW(log(2)) / log(2))
    ans = solveset_real(3 * x + 5 + 2 ** (-5 * x + 3), x)
    assert ans == FiniteSet(Rational(-5, 3) + LambertW(-10240 * 2 ** Rational(1, 3) * log(2) / 3) / (5 * log(2)))
    eq = 2 * (3 * x + 4) ** 5 - 6 * 7 ** (3 * x + 9)
    result = solveset_real(eq, x)
    ans = FiniteSet((log(2401) + 5 * LambertW(-log(7 ** (7 * 3 ** Rational(1, 5) / 5)))) / (3 * log(7)) / -1)
    assert result == ans
    assert solveset_real(eq.expand(), x) == result
    assert solveset_real(5 * x - 1 + 3 * exp(2 - 7 * x), x) == FiniteSet(Rational(1, 5) + LambertW(-21 * exp(Rational(3, 5)) / 5) / 7)
    assert solveset_real(2 * x + 5 + log(3 * x - 2), x) == FiniteSet(Rational(2, 3) + LambertW(2 * exp(Rational(-19, 3)) / 3) / 2)
    assert solveset_real(3 * x + log(4 * x), x) == FiniteSet(LambertW(Rational(3, 4)) / 3)
    assert solveset_real(x ** x - 2) == FiniteSet(exp(LambertW(log(2))))
    a = Symbol('a')
    assert solveset_real(-a * x + 2 * x * log(x), x) == FiniteSet(exp(a / 2))
    a = Symbol('a', real=True)
    assert solveset_real(a / x + exp(x / 2), x) == FiniteSet(2 * LambertW(-a / 2))
    assert solveset_real((a / x + exp(x / 2)).diff(x), x) == FiniteSet(4 * LambertW(sqrt(2) * sqrt(a) / 4))
    assert solveset_real(tanh(x + 3) * tanh(x - 3) - 1, x) is S.EmptySet
    assert solveset_real((x ** 2 - 2 * x + 1).subs(x, log(x) + 3 * x), x) == FiniteSet(LambertW(3 * S.Exp1) / 3)
    assert solveset_real((x ** 2 - 2 * x + 1).subs(x, (log(x) + 3 * x) ** 2 - 1), x) == FiniteSet(LambertW(3 * exp(-sqrt(2))) / 3, LambertW(3 * exp(sqrt(2))) / 3)
    assert solveset_real((x ** 2 - 2 * x - 2).subs(x, log(x) + 3 * x), x) == FiniteSet(LambertW(3 * exp(1 + sqrt(3))) / 3, LambertW(3 * exp(-sqrt(3) + 1)) / 3)
    assert solveset_real(x * log(x) + 3 * x + 1, x) == FiniteSet(exp(-3 + LambertW(-exp(3))))
    eq = (x * exp(x) - 3).subs(x, x * exp(x))
    assert solveset_real(eq, x) == FiniteSet(LambertW(3 * exp(-LambertW(3))))
    assert solveset_real(3 * log(a ** (3 * x + 5)) + a ** (3 * x + 5), x) == FiniteSet(-((log(a ** 5) + LambertW(Rational(1, 3))) / (3 * log(a))))
    p = symbols('p', positive=True)
    assert solveset_real(3 * log(p ** (3 * x + 5)) + p ** (3 * x + 5), x) == FiniteSet(log((-3 ** Rational(1, 3) - 3 ** Rational(5, 6) * I) * LambertW(Rational(1, 3)) ** Rational(1, 3) / (2 * p ** Rational(5, 3))) / log(p), log((-3 ** Rational(1, 3) + 3 ** Rational(5, 6) * I) * LambertW(Rational(1, 3)) ** Rational(1, 3) / (2 * p ** Rational(5, 3))) / log(p), log((3 * LambertW(Rational(1, 3)) / p ** 5) ** (1 / (3 * log(p)))))
    b = Symbol('b')
    eq = 3 * log(a ** (3 * x + 5)) + b * log(a ** (3 * x + 5)) + a ** (3 * x + 5)
    assert solveset_real(eq, x) == FiniteSet(-((log(a ** 5) + LambertW(1 / (b + 3))) / (3 * log(a))))
    assert solveset_real((a / x + exp(x / 2)).diff(x, 2), x) == FiniteSet(6 * LambertW((-1) ** Rational(1, 3) * a ** Rational(1, 3) / 3))
    assert solveset_real(x ** 3 - 3 ** x, x) == FiniteSet(-3 / log(3) * LambertW(-log(3) / 3))
    assert solveset_real(3 ** cos(x) - cos(x) ** 3) == FiniteSet(acos(-3 * LambertW(-log(3) / 3) / log(3)))
    assert solveset_real(x ** 2 - 2 ** x, x) == solveset_real(-x ** 2 + 2 ** x, x)
    assert solveset_real(3 * log(x) - x * log(3)) == FiniteSet(-3 * LambertW(-log(3) / 3) / log(3), -3 * LambertW(-log(3) / 3, -1) / log(3))
    assert solveset_real(LambertW(2 * x) - y) == FiniteSet(y * exp(y) / 2)