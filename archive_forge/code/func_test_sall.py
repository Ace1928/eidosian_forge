from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity
def test_sall():
    expr = Basic(S(1), S(2))
    expected = Basic(S(2), S(3))
    brl = sall(inc)
    assert list(brl(expr)) == [expected]
    expr = Basic(S(1), S(2), Basic(S(3), S(4)))
    expected = Basic(S(2), S(3), Basic(S(3), S(4)))
    brl = sall(do_one(inc, identity))
    assert list(brl(expr)) == [expected]