from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_printing_non_cyclic():
    p1 = Permutation([0, 1, 2, 3, 4, 5])
    assert srepr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
    assert sstr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
    p2 = Permutation([0, 1, 2])
    assert srepr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'
    assert sstr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'
    p3 = Permutation([0, 2, 1])
    assert srepr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
    assert sstr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
    p4 = Permutation([0, 1, 3, 2, 4, 5, 6, 7])
    assert srepr(p4, perm_cyclic=False) == 'Permutation([0, 1, 3, 2], size=8)'