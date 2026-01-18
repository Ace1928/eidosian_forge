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
def test_undetermined_coefficients_match():
    assert _undetermined_coefficients_match(g(x), x) == {'test': False}
    assert _undetermined_coefficients_match(sin(2 * x + sqrt(5)), x) == {'test': True, 'trialset': {cos(2 * x + sqrt(5)), sin(2 * x + sqrt(5))}}
    assert _undetermined_coefficients_match(sin(x) * cos(x), x) == {'test': False}
    s = {cos(x), x * cos(x), x ** 2 * cos(x), x ** 2 * sin(x), x * sin(x), sin(x)}
    assert _undetermined_coefficients_match(sin(x) * (x ** 2 + x + 1), x) == {'test': True, 'trialset': s}
    assert _undetermined_coefficients_match(sin(x) * x ** 2 + sin(x) * x + sin(x), x) == {'test': True, 'trialset': s}
    assert _undetermined_coefficients_match(exp(2 * x) * sin(x) * (x ** 2 + x + 1), x) == {'test': True, 'trialset': {exp(2 * x) * sin(x), x ** 2 * exp(2 * x) * sin(x), cos(x) * exp(2 * x), x ** 2 * cos(x) * exp(2 * x), x * cos(x) * exp(2 * x), x * exp(2 * x) * sin(x)}}
    assert _undetermined_coefficients_match(1 / sin(x), x) == {'test': False}
    assert _undetermined_coefficients_match(log(x), x) == {'test': False}
    assert _undetermined_coefficients_match(2 ** x * (x ** 2 + x + 1), x) == {'test': True, 'trialset': {2 ** x, x * 2 ** x, x ** 2 * 2 ** x}}
    assert _undetermined_coefficients_match(x ** y, x) == {'test': False}
    assert _undetermined_coefficients_match(exp(x) * exp(2 * x + 1), x) == {'test': True, 'trialset': {exp(1 + 3 * x)}}
    assert _undetermined_coefficients_match(sin(x) * (x ** 2 + x + 1), x) == {'test': True, 'trialset': {x * cos(x), x * sin(x), x ** 2 * cos(x), x ** 2 * sin(x), cos(x), sin(x)}}
    assert _undetermined_coefficients_match(sin(x) * (x + sin(x)), x) == {'test': False}
    assert _undetermined_coefficients_match(sin(x) * (x + sin(2 * x)), x) == {'test': False}
    assert _undetermined_coefficients_match(sin(x) * tan(x), x) == {'test': False}
    assert _undetermined_coefficients_match(x ** 2 * sin(x) * exp(x) + x * sin(x) + x, x) == {'test': True, 'trialset': {x ** 2 * cos(x) * exp(x), x, cos(x), S.One, exp(x) * sin(x), sin(x), x * exp(x) * sin(x), x * cos(x), x * cos(x) * exp(x), x * sin(x), cos(x) * exp(x), x ** 2 * exp(x) * sin(x)}}
    assert _undetermined_coefficients_match(4 * x * sin(x - 2), x) == {'trialset': {x * cos(x - 2), x * sin(x - 2), cos(x - 2), sin(x - 2)}, 'test': True}
    assert _undetermined_coefficients_match(2 ** x * x, x) == {'test': True, 'trialset': {2 ** x, x * 2 ** x}}
    assert _undetermined_coefficients_match(2 ** x * exp(2 * x), x) == {'test': True, 'trialset': {2 ** x * exp(2 * x)}}
    assert _undetermined_coefficients_match(exp(-x) / x, x) == {'test': False}
    assert _undetermined_coefficients_match(S(4), x) == {'test': True, 'trialset': {S.One}}
    assert _undetermined_coefficients_match(12 * exp(x), x) == {'test': True, 'trialset': {exp(x)}}
    assert _undetermined_coefficients_match(exp(I * x), x) == {'test': True, 'trialset': {exp(I * x)}}
    assert _undetermined_coefficients_match(sin(x), x) == {'test': True, 'trialset': {cos(x), sin(x)}}
    assert _undetermined_coefficients_match(cos(x), x) == {'test': True, 'trialset': {cos(x), sin(x)}}
    assert _undetermined_coefficients_match(8 + 6 * exp(x) + 2 * sin(x), x) == {'test': True, 'trialset': {S.One, cos(x), sin(x), exp(x)}}
    assert _undetermined_coefficients_match(x ** 2, x) == {'test': True, 'trialset': {S.One, x, x ** 2}}
    assert _undetermined_coefficients_match(9 * x * exp(x) + exp(-x), x) == {'test': True, 'trialset': {x * exp(x), exp(x), exp(-x)}}
    assert _undetermined_coefficients_match(2 * exp(2 * x) * sin(x), x) == {'test': True, 'trialset': {exp(2 * x) * sin(x), cos(x) * exp(2 * x)}}
    assert _undetermined_coefficients_match(x - sin(x), x) == {'test': True, 'trialset': {S.One, x, cos(x), sin(x)}}
    assert _undetermined_coefficients_match(x ** 2 + 2 * x, x) == {'test': True, 'trialset': {S.One, x, x ** 2}}
    assert _undetermined_coefficients_match(4 * x * sin(x), x) == {'test': True, 'trialset': {x * cos(x), x * sin(x), cos(x), sin(x)}}
    assert _undetermined_coefficients_match(x * sin(2 * x), x) == {'test': True, 'trialset': {x * cos(2 * x), x * sin(2 * x), cos(2 * x), sin(2 * x)}}
    assert _undetermined_coefficients_match(x ** 2 * exp(-x), x) == {'test': True, 'trialset': {x * exp(-x), x ** 2 * exp(-x), exp(-x)}}
    assert _undetermined_coefficients_match(2 * exp(-x) - x ** 2 * exp(-x), x) == {'test': True, 'trialset': {x * exp(-x), x ** 2 * exp(-x), exp(-x)}}
    assert _undetermined_coefficients_match(exp(-2 * x) + x ** 2, x) == {'test': True, 'trialset': {S.One, x, x ** 2, exp(-2 * x)}}
    assert _undetermined_coefficients_match(x * exp(-x), x) == {'test': True, 'trialset': {x * exp(-x), exp(-x)}}
    assert _undetermined_coefficients_match(x + exp(2 * x), x) == {'test': True, 'trialset': {S.One, x, exp(2 * x)}}
    assert _undetermined_coefficients_match(sin(x) + exp(-x), x) == {'test': True, 'trialset': {cos(x), sin(x), exp(-x)}}
    assert _undetermined_coefficients_match(exp(x), x) == {'test': True, 'trialset': {exp(x)}}
    assert _undetermined_coefficients_match(S.Half - cos(2 * x) / 2, x) == {'test': True, 'trialset': {S.One, cos(2 * x), sin(2 * x)}}
    assert _undetermined_coefficients_match(exp(2 * x) * (S.Half + cos(2 * x) / 2), x) == {'test': True, 'trialset': {exp(2 * x) * sin(2 * x), cos(2 * x) * exp(2 * x), exp(2 * x)}}
    assert _undetermined_coefficients_match(2 * x + sin(x) + cos(x), x) == {'test': True, 'trialset': {S.One, x, cos(x), sin(x)}}
    assert _undetermined_coefficients_match(cos(x) / 2 - cos(3 * x) / 2, x) == {'test': True, 'trialset': {cos(x), cos(3 * x), sin(x), sin(3 * x)}}
    assert _undetermined_coefficients_match(cos(x ** 2), x) == {'test': False}
    assert _undetermined_coefficients_match(2 ** x ** 2, x) == {'test': False}