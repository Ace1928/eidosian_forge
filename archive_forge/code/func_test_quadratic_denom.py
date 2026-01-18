from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import (Derivative, Function, diff, expand)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, csch, cosh, coth, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.polynomials import (assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.zeta_functions import polylog
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import And
from sympy.integrals.manualintegrate import (manualintegrate, find_substitutions,
from sympy.testing.pytest import raises, slow
def test_quadratic_denom():
    f = (5 * x + 2) / (3 * x ** 2 - 2 * x + 8)
    assert manualintegrate(f, x) == 5 * log(3 * x ** 2 - 2 * x + 8) / 6 + 11 * sqrt(23) * atan(3 * sqrt(23) * (x - Rational(1, 3)) / 23) / 69
    g = 3 / (2 * x ** 2 + 3 * x + 1)
    assert manualintegrate(g, x) == 3 * log(4 * x + 2) - 3 * log(4 * x + 4)