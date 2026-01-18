from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import unchanged
from sympy.core.function import (count_ops, diff, expand, expand_multinomial, Function, Derivative)
from sympy.core.mul import Mul, _keep_coeff
from sympy.core import GoldenRatio
from sympy.core.numbers import (E, Float, I, oo, pi, Rational, zoo)
from sympy.core.relational import (Eq, Lt, Gt, Ge, Le)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, csch, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan)
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.geometry.polygon import rad
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import (factor, Poly)
from sympy.simplify.simplify import (besselsimp, hypersimp, inversecombine, logcombine, nsimplify, nthroot, posify, separatevars, signsimp, simplify)
from sympy.solvers.solvers import solve
from sympy.testing.pytest import XFAIL, slow, _both_exp_pow
from sympy.abc import x, y, z, t, a, b, c, d, e, f, g, h, i, n
@_both_exp_pow
def test_separatevars():
    x, y, z, n = symbols('x,y,z,n')
    assert separatevars(2 * n * x * z + 2 * x * y * z) == 2 * x * z * (n + y)
    assert separatevars(x * z + x * y * z) == x * z * (1 + y)
    assert separatevars(pi * x * z + pi * x * y * z) == pi * x * z * (1 + y)
    assert separatevars(x * y ** 2 * sin(x) + x * sin(x) * sin(y)) == x * (sin(y) + y ** 2) * sin(x)
    assert separatevars(x * exp(x + y) + x * exp(x)) == x * (1 + exp(y)) * exp(x)
    assert separatevars((x * (y + 1)) ** z).is_Pow
    assert separatevars(1 + x + y + x * y) == (x + 1) * (y + 1)
    assert separatevars(y / pi * exp(-(z - x) / cos(n))) == y * exp(x / cos(n)) * exp(-z / cos(n)) / pi
    assert separatevars((x + y) * (x - y) + y ** 2 + 2 * x + 1) == (x + 1) ** 2
    p = Symbol('p', positive=True)
    assert separatevars(sqrt(p ** 2 + x * p ** 2)) == p * sqrt(1 + x)
    assert separatevars(sqrt(y * (p ** 2 + x * p ** 2))) == p * sqrt(y * (1 + x))
    assert separatevars(sqrt(y * (p ** 2 + x * p ** 2)), force=True) == p * sqrt(y) * sqrt(1 + x)
    assert separatevars(sqrt(x * y)).is_Pow
    assert separatevars(sqrt(x * y), force=True) == sqrt(x) * sqrt(y)
    assert separatevars((2 * x + 2) * y, dict=True, symbols=()) == {'coeff': 1, x: 2 * x + 2, y: y}
    assert separatevars((2 * x + 2) * y, dict=True, symbols=[x]) == {'coeff': y, x: 2 * x + 2}
    assert separatevars((2 * x + 2) * y, dict=True, symbols=[]) == {'coeff': 1, x: 2 * x + 2, y: y}
    assert separatevars((2 * x + 2) * y, dict=True) == {'coeff': 1, x: 2 * x + 2, y: y}
    assert separatevars((2 * x + 2) * y, dict=True, symbols=None) == {'coeff': y * (2 * x + 2)}
    assert separatevars(3, dict=True) is None
    assert separatevars(2 * x + y, dict=True, symbols=()) is None
    assert separatevars(2 * x + y, dict=True) is None
    assert separatevars(2 * x + y, dict=True, symbols=None) == {'coeff': 2 * x + y}
    n, m = symbols('n,m', commutative=False)
    assert separatevars(m + n * m) == (1 + n) * m
    assert separatevars(x + x * n) == x * (1 + n)
    f = Function('f')
    assert separatevars(f(x) + x * f(x)) == f(x) + x * f(x)
    eq = x * (1 + hyper((), (), y * z))
    assert separatevars(eq) == eq
    s = separatevars(abs(x * y))
    assert s == abs(x) * abs(y) and s.is_Mul
    z = cos(1) ** 2 + sin(1) ** 2 - 1
    a = abs(x * z)
    s = separatevars(a)
    assert not a.is_Mul and s.is_Mul and (s == abs(x) * abs(z))
    s = separatevars(abs(x * y * z))
    assert s == abs(x) * abs(y) * abs(z)
    assert separatevars(abs((x + y) / z)) == abs((x + y) / z)