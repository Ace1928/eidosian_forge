from sympy.core.expr import unchanged
from sympy.sets import (ConditionSet, Intersection, FiniteSet,
from sympy.sets.sets import SetKind
from sympy.core.function import (Function, Lambda)
from sympy.core.mod import Mod
from sympy.core.kind import NumberKind
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.sets import Interval
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_simplified_FiniteSet_in_CondSet():
    assert ConditionSet(x, And(x < 1, x > -3), FiniteSet(0, 1, 2)) == FiniteSet(0)
    assert ConditionSet(x, x < 0, FiniteSet(0, 1, 2)) == EmptySet
    assert ConditionSet(x, And(x < -3), EmptySet) == EmptySet
    y = Symbol('y')
    assert ConditionSet(x, And(x > 0), FiniteSet(-1, 0, 1, y)) == Union(FiniteSet(1), ConditionSet(x, And(x > 0), FiniteSet(y)))
    assert ConditionSet(x, Eq(Mod(x, 3), 1), FiniteSet(1, 4, 2, y)) == Union(FiniteSet(1, 4), ConditionSet(x, Eq(Mod(x, 3), 1), FiniteSet(y)))