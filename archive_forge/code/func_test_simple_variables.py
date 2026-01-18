from sympy.unify.rewrite import rewriterule
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
from sympy.strategies.rl import rebuild
from sympy.assumptions import Q
def test_simple_variables():
    rl = rewriterule(Basic(x, S(1)), Basic(x, S(2)), variables=(x,))
    assert list(rl(Basic(S(3), S(1)))) == [Basic(S(3), S(2))]
    rl = rewriterule(x ** 2, x ** 3, variables=(x,))
    assert list(rl(y ** 2)) == [y ** 3]