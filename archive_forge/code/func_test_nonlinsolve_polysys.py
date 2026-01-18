from math import isclose
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.functions.special.error_functions import (erf, erfc,
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm
from sympy.solvers import solve
from sympy.solvers.solveset import (
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
def test_nonlinsolve_polysys():
    x, y, z = symbols('x, y, z', real=True)
    assert nonlinsolve([x ** 2 + y - 2, x ** 2 + y], [x, y]) == S.EmptySet
    s = (-y + 2, y)
    assert nonlinsolve([(x + y) ** 2 - 4, x + y - 2], [x, y]) == FiniteSet(s)
    system = [x ** 2 - y ** 2]
    soln_real = FiniteSet((-y, y), (y, y))
    soln_complex = FiniteSet((-Abs(y), y), (Abs(y), y))
    soln = soln_real + soln_complex
    assert nonlinsolve(system, [x, y]) == soln
    system = [x ** 2 - y ** 2]
    soln_real = FiniteSet((y, -y), (y, y))
    soln_complex = FiniteSet((y, -Abs(y)), (y, Abs(y)))
    soln = soln_real + soln_complex
    assert nonlinsolve(system, [y, x]) == soln
    system = [x ** 2 + y - 3, x - y - 4]
    assert nonlinsolve(system, (x, y)) != nonlinsolve(system, (y, x))
    assert nonlinsolve([-x ** 2 - y ** 2 + z, -2 * x, -2 * y, S.One], [x, y, z]) == S.EmptySet
    assert nonlinsolve([x + y + z, S.One, S.One, S.One], [x, y, z]) == S.EmptySet
    system = [-x ** 2 * z ** 2 + x * y * z + y ** 4, -2 * x * z ** 2 + y * z, x * z + 4 * y ** 3, -2 * x ** 2 * z + x * y]
    assert nonlinsolve(system, [x, y, z]) == FiniteSet((0, 0, z), (x, 0, 0))