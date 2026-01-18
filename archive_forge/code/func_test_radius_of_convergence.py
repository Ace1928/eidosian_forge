from sympy.core.containers import Tuple
from sympy.core.function import Derivative
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)
from sympy.series.order import O
from sympy.abc import x, z, k
from sympy.series.limits import limit
from sympy.testing.pytest import raises, slow
from sympy.core.random import (
def test_radius_of_convergence():
    assert hyper((1, 2), [3], z).radius_of_convergence == 1
    assert hyper((1, 2), [3, 4], z).radius_of_convergence is oo
    assert hyper((1, 2, 3), [4], z).radius_of_convergence == 0
    assert hyper((0, 1, 2), [4], z).radius_of_convergence is oo
    assert hyper((-1, 1, 2), [-4], z).radius_of_convergence == 0
    assert hyper((-1, -2, 2), [-1], z).radius_of_convergence is oo
    assert hyper((-1, 2), [-1, -2], z).radius_of_convergence == 0
    assert hyper([-1, 1, 3], [-2, 2], z).radius_of_convergence == 1
    assert hyper([-1, 1], [-2, 2], z).radius_of_convergence is oo
    assert hyper([-1, 1, 3], [-2], z).radius_of_convergence == 0
    assert hyper((-1, 2, 3, 4), [], z).radius_of_convergence is oo
    assert hyper([1, 1], [3], 1).convergence_statement == True
    assert hyper([1, 1], [2], 1).convergence_statement == False
    assert hyper([1, 1], [2], -1).convergence_statement == True
    assert hyper([1, 1], [1], -1).convergence_statement == False