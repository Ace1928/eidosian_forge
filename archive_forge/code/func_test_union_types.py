from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_union_types():

    @dispatch((A, C))
    def f(x):
        return 1
    assert f(A()) == 1
    assert f(C()) == 1