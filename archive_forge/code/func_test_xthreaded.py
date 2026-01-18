from functools import wraps
from sympy.utilities.decorator import threaded, xthreaded, memoize_property, deprecated
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.core.basic import Basic
from sympy.core.relational import Eq
from sympy.matrices.dense import Matrix
from sympy.abc import x, y
def test_xthreaded():

    @xthreaded
    def function(expr, n):
        return expr ** n
    assert function(x + y, 2) == (x + y) ** 2