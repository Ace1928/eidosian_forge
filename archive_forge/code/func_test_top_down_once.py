from sympy.strategies.traverse import (
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z
def test_top_down_once():
    top_rl = top_down_once(rl)
    assert top_rl(Basic(S(1.0), S(2.0), Basic(S(3), S(4)))) == Basic2(S(1.0), S(2.0), Basic(S(3), S(4)))