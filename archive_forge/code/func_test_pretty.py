from sympy.abc import x, y
from sympy.assumptions.assume import global_assumptions
from sympy.assumptions.ask import Q
from sympy.printing import pretty
def test_pretty():
    assert pretty(Q.positive(x)) == 'Q.positive(x)'
    assert pretty({Q.positive, Q.integer}) == '{Q.integer, Q.positive}'