from sympy.concrete.summations import Sum
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.sets.sets import (FiniteSet, Interval, Union)
from sympy.solvers.inequalities import (reduce_inequalities,
from sympy.polys.rootoftools import rootof
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import solveset
from sympy.abc import x, y
from sympy.core.mod import Mod
from sympy.testing.pytest import raises, XFAIL
def test_trig_inequalities():
    assert isolve(sin(x) < S.Half, x, relational=False) == Union(Interval(0, pi / 6, False, True), Interval.open(pi * Rational(5, 6), 2 * pi))
    assert isolve(sin(x) > S.Half, x, relational=False) == Interval(pi / 6, pi * Rational(5, 6), True, True)
    assert isolve(cos(x) < S.Zero, x, relational=False) == Interval(pi / 2, pi * Rational(3, 2), True, True)
    assert isolve(cos(x) >= S.Zero, x, relational=False) == Union(Interval(0, pi / 2), Interval.Ropen(pi * Rational(3, 2), 2 * pi))
    assert isolve(tan(x) < S.One, x, relational=False) == Union(Interval.Ropen(0, pi / 4), Interval.open(pi / 2, pi))
    assert isolve(sin(x) <= S.Zero, x, relational=False) == Union(FiniteSet(S.Zero), Interval.Ropen(pi, 2 * pi))
    assert isolve(sin(x) <= S.One, x, relational=False) == S.Reals
    assert isolve(cos(x) < S(-2), x, relational=False) == S.EmptySet
    assert isolve(sin(x) >= S.NegativeOne, x, relational=False) == S.Reals
    assert isolve(cos(x) > S.One, x, relational=False) == S.EmptySet