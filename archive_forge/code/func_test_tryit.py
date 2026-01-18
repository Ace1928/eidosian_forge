from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
from io import StringIO
def test_tryit():

    def rl(expr: Basic) -> Basic:
        assert False
    safe_rl = tryit(rl, AssertionError)
    assert safe_rl(S(1)) == S(1)