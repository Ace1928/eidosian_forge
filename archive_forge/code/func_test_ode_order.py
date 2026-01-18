from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect
from sympy.solvers.ode import (classify_ode,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.utilities.misc import filldedent
def test_ode_order():
    f = Function('f')
    g = Function('g')
    x = Symbol('x')
    assert ode_order(3 * x * exp(f(x)), f(x)) == 0
    assert ode_order(x * diff(f(x), x) + 3 * x * f(x) - sin(x) / x, f(x)) == 1
    assert ode_order(x ** 2 * f(x).diff(x, x) + x * diff(f(x), x) - f(x), f(x)) == 2
    assert ode_order(diff(x * exp(f(x)), x, x), f(x)) == 2
    assert ode_order(diff(x * diff(x * exp(f(x)), x, x), x), f(x)) == 3
    assert ode_order(diff(f(x), x, x), g(x)) == 0
    assert ode_order(diff(f(x), x, x) * diff(g(x), x), f(x)) == 2
    assert ode_order(diff(f(x), x, x) * diff(g(x), x), g(x)) == 1
    assert ode_order(diff(x * diff(x * exp(f(x)), x, x), x), g(x)) == 0
    assert ode_order(Derivative(x * f(x), x), f(x)) == 1
    assert ode_order(x * sin(Derivative(x * f(x) ** 2, x, x)), f(x)) == 2
    assert ode_order(Derivative(x * Derivative(x * exp(f(x)), x, x), x), g(x)) == 0
    assert ode_order(Derivative(f(x), x, x), g(x)) == 0
    assert ode_order(Derivative(x * exp(f(x)), x, x), f(x)) == 2
    assert ode_order(Derivative(f(x), x, x) * Derivative(g(x), x), g(x)) == 1
    assert ode_order(Derivative(x * Derivative(f(x), x, x), x), f(x)) == 3
    assert ode_order(x * sin(Derivative(x * Derivative(f(x), x) ** 2, x, x)), f(x)) == 3