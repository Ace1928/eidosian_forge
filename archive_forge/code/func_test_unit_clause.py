from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_unit_clause():
    A, B, C = symbols('A,B,C')
    assert find_unit_clause([A], {}) == (A, True)
    assert find_unit_clause([A, ~A], {}) == (A, True)
    assert find_unit_clause([A | B], {A: True}) == (B, True)
    assert find_unit_clause([A | B], {B: True}) == (A, True)
    assert find_unit_clause([A | B | C, B | ~C, A | ~B], {A: True}) == (B, False)
    assert find_unit_clause([A | B | C, B | ~C, A | B], {A: True}) == (B, True)
    assert find_unit_clause([A | B | C, B | ~C, A], {}) == (A, True)