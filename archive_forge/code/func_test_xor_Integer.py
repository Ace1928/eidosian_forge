import numbers as nums
import decimal
from sympy.concrete.summations import Sum
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import (mpf_norm, mod_inverse, igcd, seterr,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.polys.domains.realfield import RealField
from sympy.printing.latex import latex
from sympy.printing.repr import srepr
from sympy.simplify import simplify
from sympy.core.power import integer_nthroot, isqrt, integer_log
from sympy.polys.domains.groundtypes import PythonRational
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
from mpmath import mpf
from mpmath.rational import mpq
import mpmath
from sympy.core import numbers
def test_xor_Integer():
    assert Integer(85) ^ Integer(255) == Integer(170)
    assert Integer(85) ^ 255 == Integer(170)
    assert 85 ^ Integer(255) == Integer(170)
    assert Integer(85) ^ Integer(219) == Integer(142)
    assert Integer(85) ^ 219 == Integer(142)
    assert 85 ^ Integer(219) == Integer(142)
    assert -Integer(85) ^ Integer(219) == Integer(-85 ^ 219)
    assert Integer(-85) ^ 219 == Integer(-85 ^ 219)
    assert -85 ^ Integer(219) == Integer(-85 ^ 219)
    assert Integer(85) ^ -Integer(219) == Integer(85 ^ -219)
    assert Integer(85) ^ -219 == Integer(85 ^ -219)
    assert 85 ^ Integer(-219) == Integer(85 ^ -219)
    raises(TypeError, lambda: Integer(2) ^ 0.0)
    raises(TypeError, lambda: 0.0 ^ Integer(2))