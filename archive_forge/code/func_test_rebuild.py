from sympy.core.singleton import S
from sympy.strategies.rl import (
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x
def test_rebuild():
    expr = Basic.__new__(Add, S(1), S(2))
    assert rebuild(expr) == 3