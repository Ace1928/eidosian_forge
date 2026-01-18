from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_permutation_apply():
    x = Symbol('x')
    p = Permutation(0, 1, 2)
    assert p.apply(0) == 1
    assert isinstance(p.apply(0), Integer)
    assert p.apply(x) == AppliedPermutation(p, x)
    assert AppliedPermutation(p, x).subs(x, 0) == 1
    x = Symbol('x', integer=False)
    raises(NotImplementedError, lambda: p.apply(x))
    x = Symbol('x', negative=True)
    raises(NotImplementedError, lambda: p.apply(x))