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
def test_to_Sequence_Initial_Coniditons():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    n = symbols('n', integer=True)
    _, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
    q = [(HolonomicSequence(-1 + (n + 1) * Sn, 1), 0)]
    assert p == q
    p = HolonomicFunction(Dx ** 2 + 1, x, 0, [0, 1]).to_sequence()
    q = [(HolonomicSequence(1 + (n ** 2 + 3 * n + 2) * Sn ** 2, [0, 1]), 0)]
    assert p == q
    p = HolonomicFunction(Dx ** 2 + 1 + x ** 3 * Dx, x, 0, [2, 3]).to_sequence()
    q = [(HolonomicSequence(n + Sn ** 2 + (n ** 2 + 7 * n + 12) * Sn ** 4, [2, 3, -1, Rational(-1, 2), Rational(1, 12)]), 1)]
    assert p == q
    p = HolonomicFunction(x ** 3 * Dx ** 5 + 1 + Dx, x).to_sequence()
    q = [(HolonomicSequence(1 + (n + 1) * Sn + (n ** 5 - 5 * n ** 3 + 4 * n) * Sn ** 2), 0, 3)]
    assert p == q
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    p = expr_to_holonomic(log(1 + x ** 2))
    q = [(HolonomicSequence(n ** 2 + (n ** 2 + 2 * n) * Sn ** 2, [0, 0, C_2]), 0, 1)]
    assert p.to_sequence() == q
    p = p.diff()
    q = [(HolonomicSequence(n + 2 + (n + 2) * Sn ** 2, [C_0, 0]), 1, 0)]
    assert p.to_sequence() == q
    p = expr_to_holonomic(erf(x) + x).to_sequence()
    q = [(HolonomicSequence(2 * n ** 2 - 2 * n + (n ** 3 + 2 * n ** 2 - n - 2) * Sn ** 2, [0, 1 + 2 / sqrt(pi), 0, C_3]), 0, 2)]
    assert p == q