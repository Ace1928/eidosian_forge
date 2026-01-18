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
def test_nthroot():
    assert nthroot(90 + 34 * sqrt(7), 3) == sqrt(7) + 3
    q = 1 + sqrt(2) - 2 * sqrt(3) + sqrt(6) + sqrt(7)
    assert nthroot(expand_multinomial(q ** 3), 3) == q
    assert nthroot(41 + 29 * sqrt(2), 5) == 1 + sqrt(2)
    assert nthroot(-41 - 29 * sqrt(2), 5) == -1 - sqrt(2)
    expr = 1320 * sqrt(10) + 4216 + 2576 * sqrt(6) + 1640 * sqrt(15)
    assert nthroot(expr, 5) == 1 + sqrt(6) + sqrt(15)
    q = 1 + sqrt(2) + sqrt(3) + sqrt(5)
    assert expand_multinomial(nthroot(expand_multinomial(q ** 5), 5)) == q
    q = 1 + sqrt(2) + 7 * sqrt(6) + 2 * sqrt(10)
    assert nthroot(expand_multinomial(q ** 5), 5, 8) == q
    q = 1 + sqrt(2) - 2 * sqrt(3) + 1171 * sqrt(6)
    assert nthroot(expand_multinomial(q ** 3), 3) == q
    assert nthroot(expand_multinomial(q ** 6), 6) == q