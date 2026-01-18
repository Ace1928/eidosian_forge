from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, true)
from sympy.matrices.common import MatrixKind
from sympy.matrices.dense import Matrix
from sympy.polys.rootoftools import rootof
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range)
from sympy.sets.sets import (Complement, DisjointUnion, FiniteSet, Intersection, Interval, ProductSet, Set, SymmetricDifference, Union, imageset, SetKind)
from mpmath import mpi
from sympy.core.expr import unchanged
from sympy.core.relational import Eq, Ne, Le, Lt, LessThan
from sympy.logic import And, Or, Xor
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.abc import x, y, z, m, n
def test_product_basic():
    H, T = ('H', 'T')
    unit_line = Interval(0, 1)
    d6 = FiniteSet(1, 2, 3, 4, 5, 6)
    d4 = FiniteSet(1, 2, 3, 4)
    coin = FiniteSet(H, T)
    square = unit_line * unit_line
    assert (0, 0) in square
    assert 0 not in square
    assert (H, T) in coin ** 2
    assert (0.5, 0.5, 0.5) in (square * unit_line).flatten()
    assert ((0.5, 0.5), 0.5) in square * unit_line
    assert (H, 3, 3) in (coin * d6 * d6).flatten()
    assert ((H, 3), 3) in coin * d6 * d6
    HH, TT = (sympify(H), sympify(T))
    assert set(coin ** 2) == {(HH, HH), (HH, TT), (TT, HH), (TT, TT)}
    assert (d4 * d4).is_subset(d6 * d6)
    assert square.complement(Interval(-oo, oo) * Interval(-oo, oo)) == Union((Interval(-oo, 0, True, True) + Interval(1, oo, True, True)) * Interval(-oo, oo), Interval(-oo, oo) * (Interval(-oo, 0, True, True) + Interval(1, oo, True, True)))
    assert (Interval(-5, 5) ** 3).is_subset(Interval(-10, 10) ** 3)
    assert not (Interval(-10, 10) ** 3).is_subset(Interval(-5, 5) ** 3)
    assert not (Interval(-5, 5) ** 2).is_subset(Interval(-10, 10) ** 3)
    assert (Interval(0.2, 0.5) * FiniteSet(0.5)).is_subset(square)
    assert len(coin * coin * coin) == 8
    assert len(S.EmptySet * S.EmptySet) == 0
    assert len(S.EmptySet * coin) == 0
    raises(TypeError, lambda: len(coin * Interval(0, 2)))