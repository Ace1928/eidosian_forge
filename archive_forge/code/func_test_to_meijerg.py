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
def test_to_meijerg():
    x = symbols('x')
    assert hyperexpand(expr_to_holonomic(sin(x)).to_meijerg()) == sin(x)
    assert hyperexpand(expr_to_holonomic(cos(x)).to_meijerg()) == cos(x)
    assert hyperexpand(expr_to_holonomic(exp(x)).to_meijerg()) == exp(x)
    assert hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify() == log(x)
    assert expr_to_holonomic(4 * x ** 2 / 3 + 7).to_meijerg() == 4 * x ** 2 / 3 + 7
    assert hyperexpand(expr_to_holonomic(besselj(2, x), lenics=3).to_meijerg()) == besselj(2, x)
    p = hyper((Rational(-1, 2), -3), (), x)
    assert from_hyper(p).to_meijerg() == hyperexpand(p)
    p = hyper((S.One, S(3)), (S(2),), x)
    assert (hyperexpand(from_hyper(p).to_meijerg()) - hyperexpand(p)).expand() == 0
    p = from_hyper(hyper((-2, -3), (S.Half,), x))
    s = hyperexpand(hyper((-2, -3), (S.Half,), x))
    C_0 = Symbol('C_0')
    C_1 = Symbol('C_1')
    D_0 = Symbol('D_0')
    assert (hyperexpand(p.to_meijerg()).subs({C_0: 1, D_0: 0}) - s).simplify() == 0
    p.y0 = {0: [1], S.Half: [0]}
    assert (hyperexpand(p.to_meijerg()) - s).simplify() == 0
    p = expr_to_holonomic(besselj(S.Half, x), initcond=False)
    assert (p.to_expr() - (D_0 * sin(x) + C_0 * cos(x) + C_1 * sin(x)) / sqrt(x)).simplify() == 0
    p = expr_to_holonomic(besselj(S.Half, x), y0={Rational(-1, 2): [sqrt(2) / sqrt(pi), sqrt(2) / sqrt(pi)]})
    assert (p.to_expr() - besselj(S.Half, x) - besselj(Rational(-1, 2), x)).simplify() == 0