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
def test_to_Sequence():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    n = symbols('n', integer=True)
    _, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    p = HolonomicFunction(x ** 2 * Dx ** 4 + x + Dx, x).to_sequence()
    q = [(HolonomicSequence(1 + (n + 2) * Sn ** 2 + (n ** 4 + 6 * n ** 3 + 11 * n ** 2 + 6 * n) * Sn ** 3), 0, 1)]
    assert p == q
    p = HolonomicFunction(x ** 2 * Dx ** 4 + x ** 3 + Dx ** 2, x).to_sequence()
    q = [(HolonomicSequence(1 + (n ** 4 + 14 * n ** 3 + 72 * n ** 2 + 163 * n + 140) * Sn ** 5), 0, 0)]
    assert p == q
    p = HolonomicFunction(x ** 3 * Dx ** 4 + 1 + Dx ** 2, x).to_sequence()
    q = [(HolonomicSequence(1 + (n ** 4 - 2 * n ** 3 - n ** 2 + 2 * n) * Sn + (n ** 2 + 3 * n + 2) * Sn ** 2), 0, 0)]
    assert p == q
    p = HolonomicFunction(3 * x ** 3 * Dx ** 4 + 2 * x * Dx + x * Dx ** 3, x).to_sequence()
    q = [(HolonomicSequence(2 * n + (3 * n ** 4 - 6 * n ** 3 - 3 * n ** 2 + 6 * n) * Sn + (n ** 3 + 3 * n ** 2 + 2 * n) * Sn ** 2), 0, 1)]
    assert p == q