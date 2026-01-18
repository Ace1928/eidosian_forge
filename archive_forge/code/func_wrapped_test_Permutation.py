from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def wrapped_test_Permutation():
    globals()['__Perm'] = globals()['Permutation']
    globals()['Permutation'] = CustomPermutation
    test_Permutation()
    globals()['Permutation'] = globals()['__Perm']
    del globals()['__Perm']