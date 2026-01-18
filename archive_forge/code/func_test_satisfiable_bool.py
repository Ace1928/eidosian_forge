from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_satisfiable_bool():
    from sympy.core.singleton import S
    assert satisfiable(true) == {true: true}
    assert satisfiable(S.true) == {true: true}
    assert satisfiable(false) is False
    assert satisfiable(S.false) is False