from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_satisfiable_non_symbols():
    x, y = symbols('x y')
    assumptions = Q.zero(x * y)
    facts = Implies(Q.zero(x * y), Q.zero(x) | Q.zero(y))
    query = ~Q.zero(x) & ~Q.zero(y)
    refutations = [{Q.zero(x): True, Q.zero(x * y): True}, {Q.zero(y): True, Q.zero(x * y): True}, {Q.zero(x): True, Q.zero(y): True, Q.zero(x * y): True}, {Q.zero(x): True, Q.zero(y): False, Q.zero(x * y): True}, {Q.zero(x): False, Q.zero(y): True, Q.zero(x * y): True}]
    assert not satisfiable(And(assumptions, facts, query), algorithm='dpll')
    assert satisfiable(And(assumptions, facts, ~query), algorithm='dpll') in refutations
    assert not satisfiable(And(assumptions, facts, query), algorithm='dpll2')
    assert satisfiable(And(assumptions, facts, ~query), algorithm='dpll2') in refutations