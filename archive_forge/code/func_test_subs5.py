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
def test_subs5():
    e = Integral(exp(-x ** 2), (x, -oo, oo))
    assert e.subs(x, 5) == e
    e = Integral(exp(-x ** 2 + y), x)
    assert e.subs(y, 5) == Integral(exp(-x ** 2 + 5), x)
    e = Integral(exp(-x ** 2 + y), (x, x))
    assert e.subs(x, 5) == Integral(exp(y - x ** 2), (x, 5))
    assert e.subs(y, 5) == Integral(exp(-x ** 2 + 5), x)
    e = Integral(exp(-x ** 2 + y), (y, -oo, oo), (x, -oo, oo))
    assert e.subs(x, 5) == e
    assert e.subs(y, 5) == e
    e = Integral(exp(-x ** 2), (x, x))
    assert e.subs(x, 5) == Integral(exp(-x ** 2), (x, 5))
    e = Integral(exp(x), x)
    assert (e.subs(x, 1) - e.subs(x, 0) - Integral(exp(x), (x, 0, 1))).doit().is_zero