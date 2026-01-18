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
def test_polylog_series():
    assert polylog(1, z).series(z, n=5) == z + z ** 2 / 2 + z ** 3 / 3 + z ** 4 / 4 + O(z ** 5)
    assert polylog(1, sqrt(z)).series(z, n=3) == z / 2 + z ** 2 / 4 + sqrt(z) + z ** (S(3) / 2) / 3 + z ** (S(5) / 2) / 5 + O(z ** 3)
    assert polylog(S(3) / 2, -z).series(z, 0, 5) == -z + sqrt(2) * z ** 2 / 4 - sqrt(3) * z ** 3 / 9 + z ** 4 / 8 + O(z ** 5)