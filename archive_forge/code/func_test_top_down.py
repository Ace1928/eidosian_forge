from sympy.strategies.traverse import (
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z
def test_top_down():
    _test_global_traversal(top_down)
    _test_stop_on_non_basics(top_down)