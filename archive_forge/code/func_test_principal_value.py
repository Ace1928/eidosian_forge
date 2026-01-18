import math
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, im, polar_lift, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (acosh, asinh, cosh, coth, csch, sinh, tanh, sech)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan, sec)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import (Ci, Ei, Si, erf, erfc, erfi, fresnelc, li)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.expr import unchanged
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import Integral
from sympy.integrals.risch import NonElementaryIntegral
from sympy.physics import units
from sympy.testing.pytest import (raises, slow, skip, ON_CI,
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
@slow
def test_principal_value():
    g = 1 / x
    assert Integral(g, (x, -oo, oo)).principal_value() == 0
    assert Integral(g, (y, -oo, oo)).principal_value() == oo * sign(1 / x)
    raises(ValueError, lambda: Integral(g, x).principal_value())
    raises(ValueError, lambda: Integral(g).principal_value())
    l = 1 / (x ** 3 - 1)
    assert Integral(l, (x, -oo, oo)).principal_value().together() == -sqrt(3) * pi / 3
    raises(ValueError, lambda: Integral(l, (x, -oo, 1)).principal_value())
    d = 1 / (x ** 2 - 1)
    assert Integral(d, (x, -oo, oo)).principal_value() == 0
    assert Integral(d, (x, -2, 2)).principal_value() == -log(3)
    v = x / (x ** 2 - 1)
    assert Integral(v, (x, -oo, oo)).principal_value() == 0
    assert Integral(v, (x, -2, 2)).principal_value() == 0
    s = x ** 2 / (x ** 2 - 1)
    assert Integral(s, (x, -oo, oo)).principal_value() is oo
    assert Integral(s, (x, -2, 2)).principal_value() == -log(3) + 4
    f = 1 / ((x ** 2 - 1) * (1 + x ** 2))
    assert Integral(f, (x, -oo, oo)).principal_value() == -pi / 2
    assert Integral(f, (x, -2, 2)).principal_value() == -atan(2) - log(3) / 2