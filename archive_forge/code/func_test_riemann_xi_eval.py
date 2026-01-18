from sympy.concrete.summations import Sum
from sympy.core.function import expand_func
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, riemann_xi, stieltjes, zeta)
from sympy.series.order import O
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
def test_riemann_xi_eval():
    assert riemann_xi(2) == pi / 6
    assert riemann_xi(0) == Rational(1, 2)
    assert riemann_xi(1) == Rational(1, 2)
    assert riemann_xi(3).rewrite(zeta) == 3 * zeta(3) / (2 * pi)
    assert riemann_xi(4) == pi ** 2 / 15