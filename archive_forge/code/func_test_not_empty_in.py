from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.functions.special.error_functions import expint
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import simplify
from sympy.calculus.util import (function_range, continuous_domain, not_empty_in,
from sympy.sets.sets import (Interval, FiniteSet, Complement, Union)
from sympy.testing.pytest import raises, _both_exp_pow
from sympy.abc import x
def test_not_empty_in():
    assert not_empty_in(FiniteSet(x, 2 * x).intersect(Interval(1, 2, True, False)), x) == Interval(S.Half, 2, True, False)
    assert not_empty_in(FiniteSet(x, x ** 2).intersect(Interval(1, 2)), x) == Union(Interval(-sqrt(2), -1), Interval(1, 2))
    assert not_empty_in(FiniteSet(x ** 2 + x, x).intersect(Interval(2, 4)), x) == Union(Interval(-sqrt(17) / 2 - S.Half, -2), Interval(1, Rational(-1, 2) + sqrt(17) / 2), Interval(2, 4))
    assert not_empty_in(FiniteSet(x / (x - 1)).intersect(S.Reals), x) == Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet(a / (a - 1)).intersect(S.Reals), a) == Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet((x ** 2 - 3 * x + 2) / (x - 1)).intersect(S.Reals), x) == Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet(3, 4, x / (x - 1)).intersect(Interval(2, 3)), x) == Interval(-oo, oo)
    assert not_empty_in(FiniteSet(4, x / (x - 1)).intersect(Interval(2, 3)), x) == Interval(S(3) / 2, 2)
    assert not_empty_in(FiniteSet(x / (x ** 2 - 1)).intersect(S.Reals), x) == Complement(S.Reals, FiniteSet(-1, 1))
    assert not_empty_in(FiniteSet(x, x ** 2).intersect(Union(Interval(1, 3, True, True), Interval(4, 5))), x) == Union(Interval(-sqrt(5), -2), Interval(-sqrt(3), -1, True, True), Interval(1, 3, True, True), Interval(4, 5))
    assert not_empty_in(FiniteSet(1).intersect(Interval(3, 4)), x) == S.EmptySet
    assert not_empty_in(FiniteSet(x ** 2 / (x + 2)).intersect(Interval(1, oo)), x) == Union(Interval(-2, -1, True, False), Interval(2, oo))
    raises(ValueError, lambda: not_empty_in(x))
    raises(ValueError, lambda: not_empty_in(Interval(0, 1), x))
    raises(NotImplementedError, lambda: not_empty_in(FiniteSet(x).intersect(S.Reals), x, a))