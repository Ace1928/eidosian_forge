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
def test_to_nnf():
    assert to_nnf(true) is true
    assert to_nnf(false) is false
    assert to_nnf(A) == A
    assert to_nnf(A | ~A | B) is true
    assert to_nnf(A & ~A & B) is false
    assert to_nnf(A >> B) == ~A | B
    assert to_nnf(Equivalent(A, B, C)) == (~A | B) & (~B | C) & (~C | A)
    assert to_nnf(A ^ B ^ C) == (A | B | C) & (~A | ~B | C) & (A | ~B | ~C) & (~A | B | ~C)
    assert to_nnf(ITE(A, B, C)) == (~A | B) & (A | C)
    assert to_nnf(Not(A | B | C)) == ~A & ~B & ~C
    assert to_nnf(Not(A & B & C)) == ~A | ~B | ~C
    assert to_nnf(Not(A >> B)) == A & ~B
    assert to_nnf(Not(Equivalent(A, B, C))) == And(Or(A, B, C), Or(~A, ~B, ~C))
    assert to_nnf(Not(A ^ B ^ C)) == (~A | B | C) & (A | ~B | C) & (A | B | ~C) & (~A | ~B | ~C)
    assert to_nnf(Not(ITE(A, B, C))) == (~A | ~B) & (A | ~C)
    assert to_nnf(A >> B ^ B >> A) == A & ~B | ~A & B
    assert to_nnf(A >> B ^ B >> A, False) == (~A | ~B | A | B) & (A & ~B | ~A & B)
    assert ITE(A, 1, 0).to_nnf() == A
    assert ITE(A, 0, 1).to_nnf() == ~A
    raises(TypeError, lambda: ITE(A < 1, [1], B).to_nnf())