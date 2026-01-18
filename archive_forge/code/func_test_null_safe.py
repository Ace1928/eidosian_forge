from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
from io import StringIO
def test_null_safe():

    def rl(expr: int) -> int | None:
        if expr == 1:
            return 2
        return None
    safe_rl = null_safe(rl)
    assert rl(1) == safe_rl(1)
    assert rl(3) is None
    assert safe_rl(3) == 3