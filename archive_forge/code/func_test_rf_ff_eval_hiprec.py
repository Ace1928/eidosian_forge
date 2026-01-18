from sympy.concrete.products import Product
from sympy.core.function import expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.factorials import subfactorial
from sympy.functions.special.gamma_functions import uppergamma
from sympy.testing.pytest import XFAIL, raises, slow
def test_rf_ff_eval_hiprec():
    maple = Float('6.9109401292234329956525265438452')
    us = ff(18, Rational(2, 3)).evalf(32)
    assert abs(us - maple) / us < 1e-31
    maple = Float('6.8261540131125511557924466355367')
    us = rf(18, Rational(2, 3)).evalf(32)
    assert abs(us - maple) / us < 1e-31
    maple = Float('34.007346127440197150854651814225')
    us = rf(Float('4.4', 32), Float('2.2', 32))
    assert abs(us - maple) / us < 1e-31