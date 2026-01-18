from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity
def test_top_down_big_tree():
    expr = Basic(S(1), Basic(S(2)), Basic(S(3), Basic(S(4)), S(5)))
    expected = Basic(S(2), Basic(S(3)), Basic(S(4), Basic(S(5)), S(6)))
    brl = top_down(inc)
    assert set(brl(expr)) == {expected}