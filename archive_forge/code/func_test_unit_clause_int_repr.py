from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_unit_clause_int_repr():
    assert find_unit_clause_int_repr(map(set, [[1]]), {}) == (1, True)
    assert find_unit_clause_int_repr(map(set, [[1], [-1]]), {}) == (1, True)
    assert find_unit_clause_int_repr([{1, 2}], {1: True}) == (2, True)
    assert find_unit_clause_int_repr([{1, 2}], {2: True}) == (1, True)
    assert find_unit_clause_int_repr(map(set, [[1, 2, 3], [2, -3], [1, -2]]), {1: True}) == (2, False)
    assert find_unit_clause_int_repr(map(set, [[1, 2, 3], [3, -3], [1, 2]]), {1: True}) == (2, True)
    A, B, C = symbols('A,B,C')
    assert find_unit_clause([A | B | C, B | ~C, A], {}) == (A, True)