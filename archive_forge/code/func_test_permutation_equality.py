from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_permutation_equality():
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, 2)
    assert Eq(a, b) is S.true
    c = Permutation(0, 2, 1)
    assert Eq(a, c) is S.false
    d = Permutation(0, 1, 2, size=4)
    assert unchanged(Eq, a, d)
    e = Permutation(0, 2, 1, size=4)
    assert unchanged(Eq, a, e)
    i = Permutation()
    assert unchanged(Eq, i, 0)
    assert unchanged(Eq, 0, i)