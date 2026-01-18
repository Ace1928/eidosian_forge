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
def test_simplification_boolalg():
    """
    Test working of simplification methods.
    """
    set1 = [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]]
    set2 = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]
    assert SOPform([x, y, z], set1) == Or(And(Not(x), z), And(Not(z), x))
    assert Not(SOPform([x, y, z], set2)) == Not(Or(And(Not(x), Not(z)), And(x, z)))
    assert POSform([x, y, z], set1 + set2) is true
    assert SOPform([x, y, z], set1 + set2) is true
    assert SOPform([Dummy(), Dummy(), Dummy()], set1 + set2) is true
    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    assert SOPform([w, x, y, z], minterms, dontcares) == Or(And(y, z), And(Not(w), Not(x)))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)
    minterms = [1, 3, 7, 11, 15]
    dontcares = [0, 2, 5]
    assert SOPform([w, x, y, z], minterms, dontcares) == Or(And(y, z), And(Not(w), Not(x)))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)
    minterms = [1, [0, 0, 1, 1], 7, [1, 0, 1, 1], [1, 1, 1, 1]]
    dontcares = [0, [0, 0, 1, 0], 5]
    assert SOPform([w, x, y, z], minterms, dontcares) == Or(And(y, z), And(Not(w), Not(x)))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)
    minterms = [1, {y: 1, z: 1}]
    dontcares = [0, [0, 0, 1, 0], 5]
    assert SOPform([w, x, y, z], minterms, dontcares) == Or(And(y, z), And(Not(w), Not(x)))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)
    minterms = [{y: 1, z: 1}, 1]
    dontcares = [[0, 0, 0, 0]]
    minterms = [[0, 0, 0]]
    raises(ValueError, lambda: SOPform([w, x, y, z], minterms))
    raises(ValueError, lambda: POSform([w, x, y, z], minterms))
    raises(TypeError, lambda: POSform([w, x, y, z], ['abcdefg']))
    ans = And(A, Or(B, C))
    assert simplify_logic(A & (B | C)) == ans
    assert simplify_logic(A & B | A & C) == ans
    assert simplify_logic(Implies(A, B)) == Or(Not(A), B)
    assert simplify_logic(Equivalent(A, B)) == Or(And(A, B), And(Not(A), Not(B)))
    assert simplify_logic(And(Equality(A, 2), C)) == And(Equality(A, 2), C)
    assert simplify_logic(And(Equality(A, 2), A)) is S.false
    assert simplify_logic(And(Equality(A, 2), A)) == And(Equality(A, 2), A)
    assert simplify_logic(And(Equality(A, B), C)) == And(Equality(A, B), C)
    assert simplify_logic(Or(And(Equality(A, 3), B), And(Equality(A, 3), C))) == And(Equality(A, 3), Or(B, C))
    b = ~x & ~y & ~z | ~x & ~y & z
    e = And(A, b)
    assert simplify_logic(e) == A & ~x & ~y
    raises(ValueError, lambda: simplify_logic(A & (B | C), form='blabla'))
    assert simplify(Or(x <= y, And(x < y, z))) == (x <= y)
    assert simplify(Or(x <= y, And(y > x, z))) == (x <= y)
    assert simplify(Or(x >= y, And(y < x, z))) == (x >= y)
    a, b, c, d, e, f, g, h, j = symbols('a b c d e f g h j')
    expr = a & b & c & d & e & f & g & h & j | a & b & c & d & e & f & g & h & ~j
    assert simplify_logic(expr) == expr
    assert simplify_logic(a & b | c | d, dontcare=a & b) == c | d
    ans = SOPform([x, y], [[1, 0]])
    assert SOPform([x, y], [[1, 0]]) == ans
    assert POSform([x, y], [[1, 0]]) == ans
    raises(ValueError, lambda: SOPform([x], [[1]], [[1]]))
    assert SOPform([x], [[1]], [[0]]) is true
    assert SOPform([x], [[0]], [[1]]) is true
    assert SOPform([x], [], []) is false
    raises(ValueError, lambda: POSform([x], [[1]], [[1]]))
    assert POSform([x], [[1]], [[0]]) is true
    assert POSform([x], [[0]], [[1]]) is true
    assert POSform([x], [], []) is false
    assert simplify(A & B | A & C) == And(A, Or(B, C))
    assert simplify(And(x, Not(x))) == False
    assert simplify(Or(x, Not(x))) == True
    assert simplify(And(Eq(x, 0), Eq(x, y))) == And(Eq(x, 0), Eq(y, 0))
    assert And(Eq(x - 1, 0), Eq(x, y)).simplify() == And(Eq(x, 1), Eq(y, 1))
    assert And(Ne(x - 1, 0), Ne(x, y)).simplify() == And(Ne(x, 1), Ne(x, y))
    assert And(Eq(x - 1, 0), Ne(x, y)).simplify() == And(Eq(x, 1), Ne(y, 1))
    assert And(Eq(x - 1, 0), Eq(x, z + y), Eq(y + x, 0)).simplify() == And(Eq(x, 1), Eq(y, -1), Eq(z, 2))
    assert And(Eq(x - 1, 0), Eq(x + 2, 3)).simplify() == Eq(x, 1)
    assert And(Ne(x - 1, 0), Ne(x + 2, 3)).simplify() == Ne(x, 1)
    assert And(Eq(x - 1, 0), Eq(x + 2, 2)).simplify() == False
    assert And(Ne(x - 1, 0), Ne(x + 2, 2)).simplify() == And(Ne(x, 1), Ne(x, 0))