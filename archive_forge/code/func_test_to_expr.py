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
def test_to_expr():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_expr()
    q = exp(x)
    assert p == q
    p = HolonomicFunction(Dx ** 2 + 1, x, 0, [1, 0]).to_expr()
    q = cos(x)
    assert p == q
    p = HolonomicFunction(Dx ** 2 - 1, x, 0, [1, 0]).to_expr()
    q = cosh(x)
    assert p == q
    p = HolonomicFunction(2 + (4 * x - 1) * Dx + (x ** 2 - x) * Dx ** 2, x, 0, [1, 2]).to_expr().expand()
    q = 1 / (x ** 2 - 2 * x + 1)
    assert p == q
    p = expr_to_holonomic(sin(x) ** 2 / x).integrate((x, 0, x)).to_expr()
    q = (sin(x) ** 2 / x).integrate((x, 0, x))
    assert p == q
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    p = expr_to_holonomic(log(1 + x ** 2)).to_expr()
    q = C_2 * log(x ** 2 + 1)
    assert p == q
    p = expr_to_holonomic(log(1 + x ** 2)).diff().to_expr()
    q = C_0 * x / (x ** 2 + 1)
    assert p == q
    p = expr_to_holonomic(erf(x) + x).to_expr()
    q = 3 * C_3 * x - 3 * sqrt(pi) * C_3 * erf(x) / 2 + x + 2 * x / sqrt(pi)
    assert p == q
    p = expr_to_holonomic(sqrt(x), x0=1).to_expr()
    assert p == sqrt(x)
    assert expr_to_holonomic(sqrt(x)).to_expr() == sqrt(x)
    p = expr_to_holonomic(sqrt(1 + x ** 2)).to_expr()
    assert p == sqrt(1 + x ** 2)
    p = expr_to_holonomic((2 * x ** 2 + 1) ** Rational(2, 3)).to_expr()
    assert p == (2 * x ** 2 + 1) ** Rational(2, 3)
    p = expr_to_holonomic(sqrt(-x ** 2 + 2 * x)).to_expr()
    assert p == sqrt(x) * sqrt(-x + 2)
    p = expr_to_holonomic((-2 * x ** 3 + 7 * x) ** Rational(2, 3)).to_expr()
    q = x ** Rational(2, 3) * (-2 * x ** 2 + 7) ** Rational(2, 3)
    assert p == q
    p = from_hyper(hyper((-2, -3), (S.Half,), x))
    s = hyperexpand(hyper((-2, -3), (S.Half,), x))
    D_0 = Symbol('D_0')
    C_0 = Symbol('C_0')
    assert (p.to_expr().subs({C_0: 1, D_0: 0}) - s).simplify() == 0
    p.y0 = {0: [1], S.Half: [0]}
    assert p.to_expr() == s
    assert expr_to_holonomic(x ** 5).to_expr() == x ** 5
    assert expr_to_holonomic(2 * x ** 3 - 3 * x ** 2).to_expr().expand() == 2 * x ** 3 - 3 * x ** 2
    a = symbols('a')
    p = (expr_to_holonomic(1.4 * x) * expr_to_holonomic(a * x, x)).to_expr()
    q = 1.4 * a * x ** 2
    assert p == q
    p = (expr_to_holonomic(1.4 * x) + expr_to_holonomic(a * x, x)).to_expr()
    q = x * (a + 1.4)
    assert p == q
    p = (expr_to_holonomic(1.4 * x) + expr_to_holonomic(x)).to_expr()
    assert p == 2.4 * x