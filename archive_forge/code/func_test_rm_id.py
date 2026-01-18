from sympy.core.singleton import S
from sympy.strategies.rl import (
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x
def test_rm_id():
    rmzeros = rm_id(lambda x: x == 0)
    assert rmzeros(Basic(S(0), S(1))) == Basic(S(1))
    assert rmzeros(Basic(S(0), S(0))) == Basic(S(0))
    assert rmzeros(Basic(S(2), S(1))) == Basic(S(2), S(1))