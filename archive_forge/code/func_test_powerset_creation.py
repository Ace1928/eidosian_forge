from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset_creation():
    assert unchanged(PowerSet, FiniteSet(1, 2))
    assert unchanged(PowerSet, S.EmptySet)
    raises(ValueError, lambda: PowerSet(123))
    assert unchanged(PowerSet, S.Reals)
    assert unchanged(PowerSet, S.Integers)