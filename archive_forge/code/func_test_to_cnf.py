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
def test_to_cnf():
    assert to_cnf(~(B | C)) == And(Not(B), Not(C))
    assert to_cnf(A & B | C) == And(Or(A, C), Or(B, C))
    assert to_cnf(A >> B) == ~A | B
    assert to_cnf(A >> (B & C)) == (~A | B) & (~A | C)
    assert to_cnf(A & (B | C) | ~A & (B | C), True) == B | C
    assert to_cnf(A & B) == And(A, B)
    assert to_cnf(Equivalent(A, B)) == And(Or(A, Not(B)), Or(B, Not(A)))
    assert to_cnf(Equivalent(A, B & C)) == (~A | B) & (~A | C) & (~B | ~C | A)
    assert to_cnf(Equivalent(A, B | C), True) == And(Or(Not(B), A), Or(Not(C), A), Or(B, C, Not(A)))
    assert to_cnf(A + 1) == A + 1