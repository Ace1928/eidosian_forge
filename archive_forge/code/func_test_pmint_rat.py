from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besselj, besselk, bessely, jn)
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.integrals.heurisch import components, heurisch, heurisch_wrapper
from sympy.testing.pytest import XFAIL, skip, slow, ON_CI
from sympy.integrals.integrals import integrate
def test_pmint_rat():

    def drop_const(expr, x):
        if expr.is_Add:
            return Add(*[arg for arg in expr.args if arg.has(x)])
        else:
            return expr
    f = (x ** 7 - 24 * x ** 4 - 4 * x ** 2 + 8 * x - 8) / (x ** 8 + 6 * x ** 6 + 12 * x ** 4 + 8 * x ** 2)
    g = (4 + 8 * x ** 2 + 6 * x + 3 * x ** 3) / (x ** 5 + 4 * x ** 3 + 4 * x) + log(x)
    assert drop_const(ratsimp(heurisch(f, x)), x) == g