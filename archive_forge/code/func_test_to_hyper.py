from sympy.holonomic import (DifferentialOperator, HolonomicFunction,
from sympy.holonomic.recurrence import RecurrenceOperators, HolonomicSequence
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (Ci, Si, erf, erfc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.simplify.hyperexpand import hyperexpand
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
def test_to_hyper():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 2, x, 0, [3]).to_hyper()
    q = 3 * hyper([], [], 2 * x)
    assert p == q
    p = hyperexpand(HolonomicFunction((1 + x) * Dx - 3, x, 0, [2]).to_hyper()).expand()
    q = 2 * x ** 3 + 6 * x ** 2 + 6 * x + 2
    assert p == q
    p = HolonomicFunction((1 + x) * Dx ** 2 + Dx, x, 0, [0, 1]).to_hyper()
    q = -x ** 2 * hyper((2, 2, 1), (3, 2), -x) / 2 + x
    assert p == q
    p = HolonomicFunction(2 * x * Dx + Dx ** 2, x, 0, [0, 2 / sqrt(pi)]).to_hyper()
    q = 2 * x * hyper((S.Half,), (Rational(3, 2),), -x ** 2) / sqrt(pi)
    assert p == q
    p = hyperexpand(HolonomicFunction(2 * x * Dx + Dx ** 2, x, 0, [1, -2 / sqrt(pi)]).to_hyper())
    q = erfc(x)
    assert p.rewrite(erfc) == q
    p = hyperexpand(HolonomicFunction(x ** 2 - 1 + x * Dx + x ** 2 * Dx ** 2, x, 0, [0, S.Half]).to_hyper())
    q = besselj(1, x)
    assert p == q
    p = hyperexpand(HolonomicFunction(x * Dx ** 2 + Dx + x, x, 0, [1, 0]).to_hyper())
    q = besselj(0, x)
    assert p == q