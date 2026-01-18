from sympy.assumptions import ask, Q
from sympy.assumptions.assume import assuming, global_assumptions
from sympy.abc import x, y
def test_remove_safe():
    global_assumptions.add(Q.integer(x))
    with assuming():
        assert ask(Q.integer(x))
        global_assumptions.remove(Q.integer(x))
        assert not ask(Q.integer(x))
    assert ask(Q.integer(x))
    global_assumptions.clear()