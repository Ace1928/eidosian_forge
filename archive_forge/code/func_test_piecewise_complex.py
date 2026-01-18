from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.function import (Function, diff, expand)
from sympy.core.mul import Mul
from sympy.core.mod import Mod
from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import (Piecewise,
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, ITE, Not, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.printing import srepr
from sympy.sets.contains import Contains
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.utilities.lambdify import lambdify
def test_piecewise_complex():
    p1 = Piecewise((2, x < 0), (1, 0 <= x))
    p2 = Piecewise((2 * I, x < 0), (I, 0 <= x))
    p3 = Piecewise((I * x, x > 1), (1 + I, True))
    p4 = Piecewise((-I * conjugate(x), x > 1), (1 - I, True))
    assert conjugate(p1) == p1
    assert conjugate(p2) == piecewise_fold(-p2)
    assert conjugate(p3) == p4
    assert p1.is_imaginary is False
    assert p1.is_real is True
    assert p2.is_imaginary is True
    assert p2.is_real is False
    assert p3.is_imaginary is None
    assert p3.is_real is None
    assert p1.as_real_imag() == (p1, 0)
    assert p2.as_real_imag() == (0, -I * p2)