from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_singledispatch():

    @dispatch(int)
    def f(x):
        return x + 1

    @dispatch(int)
    def g(x):
        return x + 2

    @dispatch(float)
    def f(x):
        return x - 1
    assert f(1) == 2
    assert g(1) == 3
    assert f(1.0) == 0
    assert raises(NotImplementedError, lambda: f('hello'))