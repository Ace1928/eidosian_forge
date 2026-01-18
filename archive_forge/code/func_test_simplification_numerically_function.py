from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import (Interval, Union)
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
from sympy.assumptions.cnf import CNF
from sympy.testing.pytest import raises, XFAIL, slow
from itertools import combinations, permutations, product
def test_simplification_numerically_function(original, simplified):
    symb = original.free_symbols
    n = len(symb)
    valuelist = list(set(combinations(list(range(-(n - 1), n)) * n, n)))
    for values in valuelist:
        sublist = dict(zip(symb, values))
        originalvalue = original.subs(sublist)
        simplifiedvalue = simplified.subs(sublist)
        assert originalvalue == simplifiedvalue, 'Original: {}\nand simplified: {}\ndo not evaluate to the same value for {}'.format(original, simplified, sublist)