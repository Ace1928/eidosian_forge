from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_torf():
    v = [T, F, U]
    for i in product(*[v] * 3):
        assert _torf(i) is (True if all((j for j in i)) else False if all((j is False for j in i)) else None)